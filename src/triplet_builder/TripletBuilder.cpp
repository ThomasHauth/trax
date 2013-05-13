/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <iomanip>
#include <set>
#include <fcntl.h>

#include <boost/program_options.hpp>

#include <clever/clever.hpp>

#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Dictionary.h>
#include <datastructures/EventSupplement.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>
#include <datastructures/LayerTriplets.h>
#include <datastructures/Pairings.h>

#include <datastructures/serialize/Event.pb.h>

#include <algorithms/PairGeneratorSector.h>
#include <algorithms/TripletThetaPhiPredictor.h>
#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/GridBuilder.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "RuntimeRecord.h"

#include "lib/ccolor.h"
#include "lib/CSV.h"

float getTIP(const Hit & p1, const Hit & p2, const Hit & p3){
	//circle fit
	//map points to parabloid: (x,y) -> (x,y,x^2+y^2)
	float3 pP1 (p1.globalX(),
			p1.globalY(),
			p1.globalX() * p1.globalX() + p1.globalY() * p1.globalY());

	float3 pP2 (p2.globalX(),
			p2.globalY(),
			p2.globalX() * p2.globalX() + p2.globalY() * p2.globalY());

	float3 pP3 (p3.globalX(),
			p3.globalY(),
			p3.globalX() * p3.globalX() + p3.globalY() * p3.globalY());

	//span two vectors
	float3 a(pP2.x - pP1.x, pP2.y - pP1.y, pP2.z - pP1.z);
	float3 b(pP3.x - pP1.x, pP3.y - pP1.y, pP3.z - pP1.z);

	//compute unit cross product
	float3 n(a.y*b.z - a.z*b.y,
			a.z*b.x - a.x*b.z,
			a.x*b.y - a.y*b.x );
	float value = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
	n.x /= value; n.y /= value; n.z /= value;

	//formula for orign and radius of circle from Strandlie et al.
	float2 cOrigin((-n.x) / (2*n.z),
			(-n.y) / (2*n.z));

	float c = -(n.x*pP1.x + n.y*pP1.y + n.z*pP1.z);

	float cR = sqrt((1 - n.z*n.z - 4 * c * n.z) / (4*n.z*n.z));

	//find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
	float2 v(-cOrigin.x, -cOrigin.y);
	value = sqrt(v.x*v.x+v.y*v.y);
	v.x /= value; v.y /= value;;

	float2 pCA = (cOrigin.x + cR*v.x,
			cOrigin.y + cR*v.y);

	//TIP = distance of point of closest approach to origin
	float tip = sqrt(pCA.x*pCA.x + pCA.y*pCA.y);

	return tip;
}

struct tEtaData{
	uint valid;
	uint fake;
	uint missed;

	tEtaData() : valid(0), fake(0), missed(0) {}
};

float getEta(const Hit & innerHit, const Hit & outerHit){
	float3 p (outerHit.globalX() - innerHit.globalX(), outerHit.globalY() - innerHit.globalY(), outerHit.globalZ() - innerHit.globalZ());

	double t(p.z/std::sqrt(p.x*p.x+p.y*p.y));
	return asinh(t);
}

float getEta(const PB_Event::PHit & innerHit, const PB_Event::PHit & outerHit){
	float3 p (outerHit.position().x() - innerHit.position().x(), outerHit.position().y() - innerHit.position().y(), outerHit.position().z() - innerHit.position().z());

	double t(p.z/std::sqrt(p.x*p.x+p.y*p.y));
	return asinh(t);
}

float getEtaBin(float eta){
	float binwidth = 0.1;
	return binwidth*floor(eta/binwidth);
}

RuntimeRecord buildTriplets(std::string filename, uint tracks, float minPt, uint threads, bool verbose = false, bool useCPU = false, int maxEvents = 1) {
	//
	std::cout << "Creating context for " << (useCPU ? "CPU" : "GPGPU") << "...";
	clever::context *contx;
	if(!useCPU){
		try{
			//try gpu
			clever::context_settings settings = clever::context_settings::default_gpu();
			settings.m_profile = true;

			contx = new clever::context(settings);
			std::cout << "success" << std::endl;
		} catch (const std::runtime_error & e){
			//if not use cpu
			clever::context_settings settings = clever::context_settings::default_cpu();
			settings.m_profile = true;
			settings.m_cmd_queue_properties = CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL;

			contx = new clever::context(settings);
			std::cout << "error: fallback on CPU" << std::endl;
		}
	} else {
		clever::context_settings settings = clever::context_settings::default_cpu();
		settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "success" << std::endl;
	}

	//load radius dictionary
	Dictionary dict;

	std::ifstream radiusDictFile("radiusDictionary.dat");
	CSVRow row;
	while(radiusDictFile >> row)
	{
		dict.addWithValue(atof(row[1].c_str()));
	}
	radiusDictFile.close();

	//load detectorGeometry
	DetectorGeometry geom;

	std::map<uint, std::pair<float, float> > exRadius; //min, max
	for(int i = 1; i <= 13; ++i){
		exRadius[i]= std::make_pair(10000.0, 0.0);
	}

	std::ifstream detectorGeometryFile("detectorRadius.dat");
	while(detectorGeometryFile >> row)
	{
		uint detId = atoi(row[0].c_str());
		uint layer = atoi(row[2].c_str());
		uint dictEntry = atoi(row[1].c_str());

		geom.addWithValue(detId, layer, dictEntry);
		DictionaryEntry entry(dict, dictEntry);

		if(entry.radius() > exRadius[layer].second) //maxiRadius
			exRadius[layer].second = entry.radius();
		if(entry.radius() < exRadius[layer].first) //minRadius
			exRadius[layer].first = entry.radius();

	}
	detectorGeometryFile.close();

	GeometrySupplement geomSupplement;
	for(auto mr : exRadius){
		geomSupplement.addWithValue(mr.first, mr.second.first, mr.second.second);
	}

	//transfer geometry to device
	geom.transfer.initBuffers(*contx, geom);
	geom.transfer.toDevice(*contx, geom);

	geomSupplement.transfer.initBuffers(*contx, geomSupplement);
	geomSupplement.transfer.toDevice(*contx, geomSupplement);

	dict.transfer.initBuffers(*contx, dict);
	dict.transfer.toDevice(*contx, dict);

	//define statistics variables
	RuntimeRecord result;
	std::map<float, tEtaData> etaHist;
	std::ofstream validTIP("validTIP.csv", std::ios::trunc);
	std::ofstream fakeTIP("fakeTIP.csv", std::ios::trunc);

	//load hit file
	PB_Event::PEventContainer pContainer;

	int fd = open(filename.c_str(), O_RDONLY);
	google::protobuf::io::FileInputStream fStream(fd);
	google::protobuf::io::CodedInputStream cStream(&fStream);

	cStream.SetTotalBytesLimit(536870912, -1);

	if(!pContainer.ParseFromCodedStream(&cStream)){
		std::cerr << "Could not read protocol buffer" << std::endl;
		return result;
	}
	cStream.~CodedInputStream();
	fStream.Close();
	fStream.~FileInputStream();
	close(fd);

	uint lastEvent;
	if(maxEvents == -1)
		lastEvent = pContainer.events_size();
	else
		lastEvent = min(maxEvents, pContainer.events_size());

	//configure hit loading
	const uint evtGrouping = 3;
	const uint maxLayer = 4;
	const uint nSectorsZ = 50;
	const uint nSectorsPhi = 8;

	const uint evtGroups = (uint) max(1.0f, ceil(((float) lastEvent )/ evtGrouping)); //number of groups

	uint event = 0;
	for(uint eventGroup = 0; eventGroup < evtGroups; ++eventGroup){

		uint evtGroupSize = std::min(evtGrouping, lastEvent-event); //if last group is not a full group

		//initialize datastructures
		EventSupplement eventSupplement(evtGroupSize);
		LayerSupplement layerSupplement(maxLayer, evtGroupSize);
		Grid grid(maxLayer, nSectorsZ,nSectorsPhi, evtGroupSize);
		HitCollection hits;
		std::map<uint, HitCollection::tTrackList> validTracks; //first: uint = evt in group second: tracklist

		do{
			uint iEvt = event % evtGroupSize;
			PB_Event::PEvent pEvent = pContainer.events(event);

			std::cout << "Started processing Event " << pEvent.eventnumber() << " LumiSection " << pEvent.lumisection() << " Run " << pEvent.runnumber() << std::endl;
			validTracks[iEvt] = hits.addEvent(pEvent, geom, eventSupplement, iEvt, layerSupplement, minPt, tracks, true, maxLayer);

			std::cout << "Loaded " << validTracks[iEvt].size() << " tracks with minPt " << minPt << " GeV and " << eventSupplement[iEvt].getNHits() << " hits" << std::endl;

			if(verbose){
				for(uint i = 1; i <= maxLayer; ++i)
					std::cout << "Layer " << i << ": " << layerSupplement[iEvt*maxLayer + i-1].getNHits() << " hits" << "\t Offset: " << layerSupplement[iEvt*maxLayer + i-1].getOffset() << std::endl;
			}

			++event;
		} while (event % evtGroupSize != 0 && event < lastEvent);

		//if(verbose)
			std::cout << "Loaded " << hits.size() << "hits in " << evtGroupSize << " event group" << std::endl;

		//transer hits to gpu
		hits.transfer.initBuffers(*contx, hits);
		hits.transfer.toDevice(*contx, hits);

		//transferring layer supplement
		eventSupplement.transfer.initBuffers(*contx, eventSupplement);
		eventSupplement.transfer.toDevice(*contx,eventSupplement);

		//transferring layer supplement
		layerSupplement.transfer.initBuffers(*contx, layerSupplement);
		layerSupplement.transfer.toDevice(*contx,layerSupplement);
		//initializating grid
		grid.transfer.initBuffers(*contx,grid);
		grid.transfer.toDevice(*contx,grid);
		grid.config.upload(*contx);

		cl_ulong runtimeBuildGrid = 0;
		GridBuilder gridBuilder(*contx);
		gridBuilder.run(hits, threads, eventSupplement, layerSupplement, grid);

		// configure kernel

		LayerTriplets layerTriplets;
		layerTriplets.addWithValue(1,2,3); //Layer Configuration
		//layerTriplets.addWithValue(2,3,4); //Layer Configuration
		layerTriplets.transfer.initBuffers(*contx, layerTriplets);
		layerTriplets.transfer.toDevice(*contx, layerTriplets);

		float dThetaCut = 0.05;
		float dThetaWindow = 0.1;
		float dPhiCut = 0.1;
		float dPhiWindow = 0.1;
		uint pairSpreadZ = 0;
		uint pairSpreadPhi = 2;
		float tipCut = 0.75;

		//run it
		PairGeneratorSector pairGen(*contx);
		Pairing  * pairs = pairGen.run(hits, threads, layerTriplets, grid, pairSpreadZ, pairSpreadPhi);

		TripletThetaPhiPredictor predictor(*contx);
		Pairing * tripletCandidates = predictor.run(hits, geom, geomSupplement, dict, threads, layerTriplets, grid, dThetaWindow, dPhiWindow, *pairs);

		TripletThetaPhiFilter tripletThetaPhi(*contx);
		TrackletCollection * tracklets = tripletThetaPhi.run(hits, grid, *pairs, *tripletCandidates, threads, layerTriplets, dThetaCut, dPhiCut, tipCut);

		//evaluate it

		for(uint iEvt = 0; iEvt < eventGroup; ++iEvt){

			std::set<uint> foundTracks;
			uint fakeTracks = 0;

			std::cout << "Found " << tracklets->size() << " triplets:" << std::endl;
			for(uint i = 0; i < tracklets->size(); ++i){
				Tracklet tracklet(*tracklets, i);

				if(tracklet.isValid(hits)){
					//valid triplet
					foundTracks.insert(tracklet.trackId(hits));

					validTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
					etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].valid++;
					if(verbose){
						std::cout << zkr::cc::fore::green;
						std::cout << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
						std::cout << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
						std::cout << zkr::cc::console << std::endl;
					}
				}
				else {
					//fake triplet
					++fakeTracks;

					fakeTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
					etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].fake++;
					if(verbose){
						std::cout << zkr::cc::fore::red;
						std::cout << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
						std::cout << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
						std::cout << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
						std::cout << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
						std::cout << zkr::cc::console << std::endl;
					}
				}
			}

			//output not found tracks
			for(auto vTrack : validTracks[iEvt]) {
				if( foundTracks.find(vTrack.first) == foundTracks.end()){
					std::cout << "Didn't find track " << vTrack.first << std::endl;

					PB_Event::PHit innerHit = vTrack.second[0];
					PB_Event::PHit outerHit = vTrack.second[vTrack.second.size()-1];

					etaHist[getEtaBin(getEta(innerHit, outerHit))].missed++;
				}
			}

			std::cout << "Efficiency: " << ((double) foundTracks.size()) / validTracks[iEvt].size() << " FakeRate: " << ((double) fakeTracks) / tracklets->size() << std::endl;

			RuntimeRecord tmpRes;
			tmpRes.nTracks = foundTracks.size();
			tmpRes.efficiency =  ((double) foundTracks.size()) / validTracks[iEvt].size();
			tmpRes.fakeRate = ((double) fakeTracks) / tracklets->size();

			//determine runtimes
			tmpRes.buildGrid = runtimeBuildGrid;
			std::cout << "Build Grid: " << runtimeBuildGrid << "ns" << std::endl;

			profile_info pinfo = contx->report_profile(contx->PROFILE_WRITE);
			tmpRes.dataTransferWrite = pinfo.runtime();
			std::cout << "Data Transfer\tWritten: " << pinfo.runtime() << "ns\tRead: ";
			pinfo = contx->report_profile(contx->PROFILE_READ);
			tmpRes.dataTransferRead = pinfo.runtime();
			std::cout << pinfo.runtime() << " ns" << std::endl;


			pinfo = contx->report_profile(PairGeneratorSector::KERNEL_COMPUTE_EVT());
			std::cout << "Pair Generation\tCompute: " << pinfo.runtime() << " ns\tStore: ";
			tmpRes.pairGenComp = pinfo.runtime();
			pinfo = contx->report_profile(PairGeneratorSector::KERNEL_STORE_EVT());
			tmpRes.pairGenStore = pinfo.runtime();
			std::cout << pinfo.runtime() << " ns" << std::endl;


			pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_COMPUTE_EVT());
			tmpRes.tripletPredictComp = pinfo.runtime();
			std::cout << "Triplet Prediction\tCompute: " << pinfo.runtime() << " ns\tStore: ";
			pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_STORE_EVT());
			tmpRes.tripletPredictStore = pinfo.runtime();
			std::cout << pinfo.runtime() << " ns" << std::endl;


			pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_COMPUTE_EVT());
			tmpRes.tripletCheckComp = pinfo.runtime();
			std::cout << "Triplet Checking\tCompute: " << pinfo.runtime() << " ns\tStore: ";
			pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_STORE_EVT());
			tmpRes.tripletCheckStore = pinfo.runtime();
			std::cout << pinfo.runtime() << " ns" << std::endl;

			result += tmpRes;
		}

		delete pairs;
		delete tripletCandidates;
		delete tracklets;

	}

	delete contx;

	validTIP.close();
	fakeTIP.close();

	std::ofstream etaData("etaData.csv", std::ios::trunc);

	etaData << "#etaBin, valid, fake, missed" << std::endl;
	for(auto t : etaHist){
		etaData << t.first << "," << t.second.valid << "," << t.second.fake << "," << t.second.missed << std::endl;
	}

	etaData.close();

	return result;
}

int main(int argc, char *argv[]) {

	namespace po = boost::program_options;

	float minPt;
	uint tracks;
	uint threads;
	bool silent;
	bool verbose;
	bool useCPU;
	int maxEvents;
	std::string srcFile;
	std::string configFile;

	po::options_description config("Configuration");
	config.add_options()
		("minPt", po::value<float>(&minPt)->default_value(1.0), "minimum track Pt")
		("tracks", po::value<uint>(&tracks)->default_value(100), "number of valid tracks to load")
		("threads", po::value<uint>(&threads)->default_value(4), "number of threads to use")
		("maxEvents", po::value<int>(&maxEvents)->default_value(1), "number of events to process")
		("src", po::value<std::string>(&srcFile)->default_value("hits_ttbar_PXB.sim.pb"), "hit database");

	po::options_description cmdLine("Generic Options");
	cmdLine.add(config);
	cmdLine.add_options()
			("help", "produce help message")
			("silent", po::value<bool>(&silent)->default_value(false)->zero_tokens(), "supress all messages from TripletFinder")
			("verbose", po::value<bool>(&verbose)->default_value(false)->zero_tokens(), "elaborate information")
			("use-cpu", po::value<bool>(&useCPU)->default_value(false)->zero_tokens(), "force using CPU instead of GPU")
			("testSuite", "run entire testSuite")
			("config", po::value<std::string>(&configFile), "config file");

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,cmdLine), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << cmdLine << std::endl;
		return 1;
	}

	if(vm.count("config")){
		ifstream ifs(configFile);
		if (!ifs)
		{
			cout << "can not open config file: " << configFile << "\n";
			return 0;
		}
		else
		{
			store(parse_config_file(ifs, config), vm);
			notify(vm);
		}
	}

	if(vm.count("testSuite")){

		uint testCases[] = {1, 10, 50, 100, 200, 300, 500 };

		std::ofstream results("timings.csv", std::ios::trunc);

		results << "#nTracks, dataTransfer, buildGrid, pairGen, tripletPredict, tripletFilter, computation, runtime, efficiency, fakeRate" << std::endl;

		for(uint i : testCases){
			RuntimeRecord res = buildTriplets(srcFile,i,minPt, threads, useCPU, maxEvents);

			results << i << ", " << res.totalDataTransfer() << ", " << res.buildGrid << ", " << res.totalPairGen() << ", " << res.totalTripletPredict() << ", " << res.totalTripletCheck() << ", "  << res.totalComputation() << ", " << res.totalRuntime()
						<< ", " << res.efficiency << ", " << res.fakeRate << std::endl;
		}

		results.close();

		return 0;
	}

	std::streambuf * coutSave = NULL;
	if(silent){
		std::ofstream devNull("/dev/null");
		coutSave = std::cout.rdbuf();
		std::cout.rdbuf(devNull.rdbuf());
	}

	RuntimeRecord res = buildTriplets(srcFile, tracks,minPt, threads, verbose, useCPU, maxEvents);

	std::ofstream results("timings.csv", std::ios::app);
        results << tracks << ", " << res.totalDataTransfer() << ", " << res.buildGrid << ", " 
                << res.totalPairGen() << ", " << res.totalTripletPredict() << ", " << res.totalTripletCheck() << ", "  
                << res.totalComputation() << ", " << res.totalRuntime()
                << ", " << res.efficiency << ", " << res.fakeRate << std::endl;
        results.close();


	if(silent){
		std::cout.rdbuf(coutSave);
	}

	std::cout << "Found: " << res.nTracks << " Tracks with mintPt=" << minPt << " using "
			<< threads << " threads in " << res.totalRuntime() << " ns" << std::endl;
	std::cout << "\tData transfer " << res.totalDataTransfer() << " ns" << std::endl;
	std::cout << "\tBuild grid " << res.buildGrid << " ns" << std::endl;
	std::cout << "\tPairGen "	<< res.totalPairGen() << " ns" << std::endl;
	std::cout << "\tTripletPredict " << res.totalTripletPredict() << " ns" << std::endl;
	std::cout << "\tTripletCheck " << res.totalTripletCheck() << " ns" << std::endl;
	std::cout << "\tTotal Computation "	<< res.totalComputation() << " ns" << std::endl;

}
