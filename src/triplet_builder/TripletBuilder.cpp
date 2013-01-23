/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <set>

#include <boost/program_options.hpp>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.cpp>
#include <datastructures/Dictionary.h>
#include <datastructures/LayerSupplement.h>

#include <algorithms/TripletThetaPhiFilter.h>

#include "RuntimeRecord.h"

#include "lib/ccolor.cpp"
#include "lib/CSV.h"

RuntimeRecord buildTriplets(uint tracks, float minPt, uint threads) {
	//
	clever::context *contx;
	try{
		//try gpu
		clever::context_settings settings = clever::context_settings::default_gpu();
		settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "Using GPGPU" << std::endl;
	} catch (const std::runtime_error & e){
		//if not use cpu
		clever::context_settings settings = clever::context_settings::default_cpu();
		settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "Using CPU" << std::endl;
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

	std::ifstream detectorGeometryFile("detectorRadius.dat");
	while(detectorGeometryFile >> row)
	{
		geom.addWithValue(atoi(row[0].c_str()), atoi(row[1].c_str()));
	}
	detectorGeometryFile.close();

	//configure hit loader
	const int maxLayer = 3;
	const int nSectors = 8;
	LayerSupplement layerSupplement(maxLayer, nSectors);

	HitCollection hits;
	HitCollection::tTrackList validTracks = HitCollectionData::loadHitDataFromPB(hits, "hitsPXB.pb", geom, layerSupplement, nSectors, minPt, tracks,true, maxLayer);

	std::cout << "Loaded " << validTracks.size() << " tracks with minPt " << minPt << " GeV and " << hits.size() << " hits" << std::endl;
	for(int i = 1; i <= maxLayer; ++i)
		std::cout << "Layer " << i << ": " << layerSupplement[i-1].nHits << " hits" << std::endl;

	//output sector borders
	std::cout << "Layer";
	for(int i = 0; i <= nSectors; ++i){
		std::cout << "\t" << -M_PI + i * (2*M_PI / nSectors);
	}
	std::cout << std::endl;

	for(int i = 0; i < maxLayer; i++){
		std::cout << i+1;
		for(int j = 0; j <= nSectors; j++){
			std::cout << "\t" << layerSupplement[i].sectorBorders[j];
		}
		std::cout << std::endl;
	}

	//transer everything to gpu
	HitCollectionTransfer hitTransfer;
	hitTransfer.initBuffers(*contx, hits);
	hitTransfer.toDevice(*contx, hits);

	DetectorGeometryTransfer geomTransfer;
	geomTransfer.initBuffers(*contx, geom);
	geomTransfer.toDevice(*contx, geom);

	DictionaryTransfer dictTransfer;
	dictTransfer.initBuffers(*contx, dict);
	dictTransfer.toDevice(*contx, dict);

	// configure kernel

	int layers[] = {1,2,3};

	float dTheta = 0.01;
	float dPhi = 0.1;

	//run it
	TripletThetaPhiFilter tripletThetaPhi(*contx);
	TrackletCollection * tracklets = tripletThetaPhi.run(hitTransfer, geomTransfer, dictTransfer, threads, layers,
			layerSupplement, dTheta, dPhi, nSectors);

	//evaluate it
	std::set<uint> foundTracks;
	uint fakeTracks = 0;

	std::cout << "Found " << tracklets->size() << " triplets:" << std::endl;
	for(uint i = 0; i < tracklets->size(); ++i){
		Tracklet tracklet(*tracklets, i);

		if(tracklet.isValid(hits)){
			//valid triplet
			foundTracks.insert(hits.getValue(HitId(),tracklet.hit1()));
			std::cout << zkr::cc::fore::green;
			std::cout << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
			std::cout << zkr::cc::console << std::endl;
		}
		else {
			//fake triplet
			++fakeTracks;
			foundTracks.insert(hits.getValue(HitId(),tracklet.hit1()));
			std::cout << zkr::cc::fore::red;
			std::cout << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
			std::cout << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
			std::cout << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
			std::cout << zkr::cc::console << std::endl;
		}
	}

	std::cout << "Efficiency: " << ((double) foundTracks.size()) / validTracks.size() << " FakeRate: " << ((double) fakeTracks) / tracklets->size() << std::endl;

	RuntimeRecord result;
	result.nTracks = foundTracks.size();
	result.efficiency =  ((double) foundTracks.size()) / validTracks.size();
	result.fakeRate = ((double) fakeTracks) / tracklets->size();

	//output not found tracks
	for(auto vTrack : validTracks) {
		if( foundTracks.find(vTrack.first) == foundTracks.end())
			std::cout << "Didn't find track " << vTrack.first << std::endl;
	}

	//determine runtimes
	profile_info pinfo = contx->report_profile(contx->PROFILE_WRITE);
	result.dataTransferWrite = pinfo.runtime();
	std::cout << "Data Transfer\tWritten: " << pinfo.runtime() << "ns\tRead: ";
	pinfo = contx->report_profile(contx->PROFILE_READ);
	result.dataTransferRead = pinfo.runtime();
	std::cout << pinfo.runtime() << " ns" << std::endl;


	/*
	pinfo = contx->report_profile(PairGeneratorSector::KERNEL_COMPUTE_EVT());
	std::cout << "Pair Generation\tCompute: " << pinfo.runtime() << " ns\tStore: ";
	pinfo = contx->report_profile(PairGeneratorSector::KERNEL_STORE_EVT());
	std::cout << pinfo.runtime() << " ns" << std::endl;
	*/


	pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_COMPUTE_EVT());
	result.tripletPredictComp = pinfo.runtime();
	std::cout << "Triplet Prediction\tCompute: " << pinfo.runtime() << " ns\tStore: ";
	pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_STORE_EVT());
	result.tripletPredictStore = pinfo.runtime();
	std::cout << pinfo.runtime() << " ns" << std::endl;


	pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_COMPUTE_EVT());
	result.tripletCheckComp = pinfo.runtime();
	std::cout << "Triplet Checking\tCompute: " << pinfo.runtime() << " ns\tStore: ";
	pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_STORE_EVT());
	result.tripletCheckStore = pinfo.runtime();
	std::cout << pinfo.runtime() << " ns" << std::endl;

	delete tracklets;
	delete contx;

	return result;
}

int main(int argc, char *argv[]) {

	namespace po = boost::program_options;

	float minPt;
	uint tracks;
	uint threads;
	bool silent;

	po::options_description desc("Allowed Options");
	desc.add_options()
			("help", "produce help message")
			("minPt", po::value<float>(&minPt)->default_value(1.0), "minimum track Pt")
			("tracks", po::value<uint>(&tracks)->default_value(10), "number of valid tracks to load")
			("threads", po::value<uint>(&threads)->default_value(4), "number of threads to use")
			("silent", po::value<bool>(&silent)->zero_tokens(), "supress messages")
			("testSuite", "run entire testSuite");

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << desc << std::endl;
		return 1;
	}

	if(vm.count("testSuite")){

		uint testCases[] = {1, 10, 50, 100, 200, 300, 500 };

		std::ofstream results("timings.csv", std::ios::trunc);

		results << "#nTracks, dataTransfer, pairGen, tripletPredict, tripletFilter, computation, runtime, efficiency, fakeRate" << std::endl;

		for(uint i : testCases){
			RuntimeRecord res = buildTriplets(i,minPt, threads);

			results << res.nTracks << ", " << res.totalDataTransfer() << ", " << res.totalPairGen() << ", " << res.totalTripletPredict() << ", " << res.totalTripletCheck() << ", "  << res.totalComputation() << ", " << res.totalRuntime()
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

	RuntimeRecord res = buildTriplets(tracks,minPt, threads);

	if(silent){
		std::cout.rdbuf(coutSave);
	}

	std::cout << "Found: " << res.nTracks << " Tracks with mintPt=" << minPt << " using "
			<< threads << " threads in " << res.totalRuntime() << " ns" << std::endl;
	std::cout << "\tData transfer " << res.totalDataTransfer() << " ns" << std::endl;
	std::cout << "\tPairGen "	<< res.totalPairGen() << " ns" << std::endl;
	std::cout << "\tTripletPredict " << res.totalTripletPredict() << " ns" << std::endl;
	std::cout << "\tTripletCheck " << res.totalTripletCheck() << " ns" << std::endl;
	std::cout << "\tTotal Computation "	<< res.totalComputation() << " ns" << std::endl;

}
