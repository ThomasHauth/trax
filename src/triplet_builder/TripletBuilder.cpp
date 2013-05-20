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
#include <datastructures/TripletConfiguration.h>
#include <datastructures/Pairings.h>
#include <datastructures/Logger.h>

#include <datastructures/serialize/Event.pb.h>

#include <algorithms/PairGeneratorSector.h>
#include <algorithms/TripletThetaPhiPredictor.h>
#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/GridBuilder.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "RuntimeRecord.h"
#include "Parameters.h"
#include "EventLoader.h"

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

RuntimeRecords buildTriplets(ExecutionParameters exec, EventDataLoadingParameters loader, GridConfig gridConfig) {

	//set up logger verbosity
	Logger::getInstance().setLogLevel(exec.verbosity);

	//
	LOG << "Creating context for " << (exec.useCPU ? "CPU" : "GPGPU") << "...";

#ifndef CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL
#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL 0
#endif

	clever::context *contx;
	if(!exec.useCPU){
		try{
			//try gpu
			clever::context_settings settings = clever::context_settings::default_gpu();
			settings.m_profile = true;

			contx = new clever::context(settings);
			LOG << "success" << std::endl;
		} catch (const std::runtime_error & e){
			//if not use cpu
			clever::context_settings settings = clever::context_settings::default_cpu();
			settings.m_profile = true;
			settings.m_cmd_queue_properties |= CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL;

			contx = new clever::context(settings);
			LOG << "error: fallback on CPU" << std::endl;
		}
	} else {
		clever::context_settings settings = clever::context_settings::default_cpu();
		settings.m_profile = true;
		settings.m_cmd_queue_properties |= CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL;

		contx = new clever::context(settings);
		LOG << "success" << std::endl;
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
	RuntimeRecords runtimeRecords;
	std::map<float, tEtaData> etaHist;
	std::ofstream validTIP("validTIP.csv", std::ios::trunc);
	std::ofstream fakeTIP("fakeTIP.csv", std::ios::trunc);

	//Event Data Loader
	EventLoader * edLoader;
	if(!loader.singleEventLoader)
		edLoader = new EventLoader(loader);
	else
		edLoader = new RepeatedEventLoader(loader);

	uint lastEvent;
	if(loader.maxEvents == -1)
		lastEvent =edLoader->nEvents();
	else
		lastEvent = min(loader.maxEvents, edLoader->nEvents());

	TripletConfigurations layerConfig;
	loader.maxLayer = layerConfig.loadTripletConfigurationFromFile("layerTriplets.xml", 1); //TODO one defined for testing

	layerConfig.transfer.initBuffers(*contx, layerConfig);
	layerConfig.transfer.toDevice(*contx, layerConfig);

	LOG << "Loaded " << layerConfig.size() << " layer triplet configurations" << std::endl;

	if(VERBOSE){
		for(uint i = 0; i < layerConfig.size(); ++ i){
			TripletConfiguration config(layerConfig, i);
			VLOG << "Layers " << config.layer1() << "-" << config.layer2() << "-" << config.layer3() << std::endl;
			VLOG << "\t" << "dThetaCut: " << config.getValue<dThetaCut>() << " dThetaWindow: " << config.getValue<dThetaWindow>() << std::endl;
			VLOG << "\t" << "dPhiCut: " << config.getValue<dPhiCut>() << " dTPhiWindow: " << config.getValue<dPhiWindow>() << std::endl;
			VLOG << "\t" << "tipCut: " << config.getValue<tipCut>() << std::endl;
			VLOG << "\t" << "pairSpreadZ: " << config.getValue<pairSpreadZ>() << " pairSpreadPhi: " << config.getValue<pairSpreadPhi>() << std::endl;
		}
	}

	//configure hit loading

	const uint evtGroups = (uint) max(1.0f, ceil(((float) lastEvent )/ exec.eventGrouping)); //number of groups

	uint event = 0;
	for(uint eventGroup = 0; eventGroup < evtGroups; ++eventGroup){

		uint evtGroupSize = std::min(exec.eventGrouping, lastEvent-event); //if last group is not a full group

		//initialize datastructures
		EventSupplement eventSupplement(evtGroupSize);
		LayerSupplement layerSupplement(loader.maxLayer, evtGroupSize);

		gridConfig.nLayers = loader.maxLayer;
		gridConfig.nEvents = evtGroupSize;
		Grid grid(gridConfig);

		HitCollection hits;
		std::map<uint, HitCollection::tTrackList> validTracks; //first: uint = evt in group second: tracklist
		uint totalValidTracks = 0;

		do{
			uint iEvt = event % evtGroupSize;
			PB_Event::PEvent pEvent = edLoader->getEvent(event);

			LOG << "Started processing Event " << pEvent.eventnumber() << " LumiSection " << pEvent.lumisection() << " Run " << pEvent.runnumber() << std::endl;
			validTracks[iEvt] = hits.addEvent(pEvent, geom, eventSupplement, iEvt, layerSupplement,
					loader.minPt, loader.maxTracks, loader.onlyTracks, loader.maxLayer);

			totalValidTracks += validTracks[iEvt].size();
			LOG << "Loaded " << validTracks[iEvt].size() << " tracks with minPt " << loader.minPt << " GeV and " << eventSupplement[iEvt].getNHits() << " hits" << std::endl;

			if(VERBOSE){
				for(uint i = 1; i <= loader.maxLayer; ++i)
					VLOG << "Layer " << i << ": " << layerSupplement[iEvt*loader.maxLayer + i-1].getNHits() << " hits" << "\t Offset: " << layerSupplement[iEvt*loader.maxLayer + i-1].getOffset() << std::endl;
			}

			++event;
		} while (event % evtGroupSize != 0 && event < lastEvent);

		LOG << "Loaded " << hits.size() << " hits in " << evtGroupSize << " events" << std::endl;

		RuntimeRecord runtime(grid.config.nEvents, grid.config.nLayers, layerConfig.size(), hits.size(), totalValidTracks, exec.threads);

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

		GridBuilder gridBuilder(*contx);

		runtime.buildGrid.startWalltime();
		gridBuilder.run(hits, exec.threads, eventSupplement, layerSupplement, grid);
		runtime.buildGrid.stopWalltime();

		//run it
		PairGeneratorSector pairGen(*contx);

		runtime.pairGen.startWalltime();
		Pairing  * pairs = pairGen.run(hits, exec.threads, layerConfig, grid);
		runtime.pairGen.stopWalltime();

		TripletThetaPhiPredictor predictor(*contx);

		runtime.tripletPredict.startWalltime();
		Pairing * tripletCandidates = predictor.run(hits, geom, geomSupplement, dict, exec.threads, layerConfig, grid, *pairs);
		runtime.tripletPredict.stopWalltime();

		TripletThetaPhiFilter tripletThetaPhi(*contx);

		runtime.tripletFilter.startWalltime();
		TrackletCollection * tracklets = tripletThetaPhi.run(hits, grid, *pairs, *tripletCandidates, exec.threads, layerConfig);
		runtime.tripletFilter.stopWalltime();

		//evaluate it

		//runtime
		runtime.fillRuntimes(*contx);
		runtime.logPrint();
		runtimeRecords.addRecord(runtime);

		//physics
		for(uint e = 0; e < grid.config.nEvents; ++e){

			for(uint p = 0; p < layerConfig.size(); ++ p){

				LOG << "Evaluating event " << e << " layer triplet " << p  << std::endl;

				std::set<uint> foundTracks;
				uint fakeTracks = 0;

				uint nFoundTracklets = tracklets->getTrackletOffsets()[e * layerConfig.size() + p + 1] - tracklets->getTrackletOffsets()[e * layerConfig.size() + p];

				LOG << "Found " << nFoundTracklets << " triplets:" << std::endl;
				for(uint i = tracklets->getTrackletOffsets()[e * layerConfig.size() + p]; i <tracklets->getTrackletOffsets()[e * layerConfig.size() + p + 1]; ++i){
					Tracklet tracklet(*tracklets, i);

					if(tracklet.isValid(hits)){
						//valid triplet
						foundTracks.insert(tracklet.trackId(hits));

						validTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
						etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].valid++;

							VLOG << zkr::cc::fore::green;
							VLOG << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
							VLOG << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
							VLOG << zkr::cc::console << std::endl;

					}
					else {
						//fake triplet
						++fakeTracks;

						fakeTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
						etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].fake++;

							VLOG << zkr::cc::fore::red;
							VLOG << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
							VLOG << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
							VLOG << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
							VLOG << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
							VLOG << zkr::cc::console << std::endl;

					}
				}

				//output not found tracks
				for(auto vTrack : validTracks[e]) {
					if( foundTracks.find(vTrack.first) == foundTracks.end()){
						VLOG << "Didn't find track " << vTrack.first << std::endl;

						PB_Event::PHit innerHit = vTrack.second[0];
						PB_Event::PHit outerHit = vTrack.second[vTrack.second.size()-1];

						etaHist[getEtaBin(getEta(innerHit, outerHit))].missed++;
					}
				}

				LOG << "Efficiency: " << ((double) foundTracks.size()) / validTracks[e].size() << " FakeRate: " << ((double) fakeTracks) / tracklets->size() << std::endl;

			}
		}

		//delete variables
		delete pairs;
		delete tripletCandidates;
		delete tracklets;

		//reset kernels event counts
		GridBuilder::clearEvents();
		PairGeneratorSector::clearEvents();
		TripletThetaPhiPredictor::clearEvents();
		TripletThetaPhiFilter::clearEvents();
		PrefixSum::clearEvents();

	}

	delete contx;
	delete edLoader;

	validTIP.close();
	fakeTIP.close();

	std::ofstream etaData("etaData.csv", std::ios::trunc);

	etaData << "#etaBin, valid, fake, missed" << std::endl;
	for(auto t : etaHist){
		etaData << t.first << "," << t.second.valid << "," << t.second.fake << "," << t.second.missed << std::endl;
	}

	etaData.close();

	return runtimeRecords;
}

int main(int argc, char *argv[]) {

	namespace po = boost::program_options;

	ExecutionParameters exec;
	EventDataLoadingParameters loader;
	GridConfig grid;

	std::string testSuiteFile;

	po::options_description cLoader("Config File: Event Data Loading Options");
	cLoader.add_options()
		("data.edSrc", po::value<std::string>(&loader.eventDataFile), "event database")
		("data.maxEvents", po::value<int>(&loader.maxEvents)->default_value(1), "number of events to process, all events: -1")
		("data.minPt", po::value<float>(&loader.minPt)->default_value(1.0), "MC data only: minimum track Pt")
		("data.tracks", po::value<int>(&loader.maxTracks)->default_value(-1), "MC data only: number of valid tracks to load, all tracks: -1")
		("data.onlyTracks", po::value<bool>(&loader.onlyTracks)->default_value(true), "MC data only: load only tracks with hits in each layer")
		("data.singleEventLoader", po::value<bool>(&loader.singleEventLoader)->default_value(false), "only load a single event and use it maxEvents times")
	;

	po::options_description cExec("Config File: Execution Options");
	cExec.add_options()
		("exec.threads", po::value<uint>(&exec.threads)->default_value(256), "number of work-items in one work-group")
		("exec.eventGrouping", po::value<uint>(&exec.eventGrouping)->default_value(1), "number of concurrently processed events")
		("exec.useCPU", po::value<bool>(&exec.useCPU)->default_value(false)->zero_tokens(), "force using CPU instead of GPGPU")
		("exec.iterations", po::value<uint>(&exec.iterations)->default_value(1), "number of iterations for performance evaluation")
		("exec.layerTripletConfig", po::value<std::string>(&exec.layerTripletConfigFile), "configuration file for layer triplets")
	;

	po::options_description cGrid("Config File: Grid Options");
	cGrid.add_options()
		("grid.nSectorsZ", po::value<uint>(&grid.nSectorsZ)->default_value(50), "number of grid cells in z dimension")
		("grid.nSectorsPhi", po::value<uint>(&grid.nSectorsPhi)->default_value(8), "number of grid cells in phi dimension")
	;

	po::options_description cCommandLine("Command Line Options");
	cCommandLine.add_options()
		("config", po::value<std::string>(&exec.configFile), "configuration file")
		("help", "produce help message")
		("silent", "supress all messages from TripletFinder")
		("verbose", "elaborate information")
		("prolix", "prolix output -- degrades performance as data is transferred from device")
		("live", "live output -- degrades performance")
		("testSuite", po::value<std::string>(&testSuiteFile), "specify a file defining test cases to be run")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,cCommandLine), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << cCommandLine << cExec << cGrid << cLoader << std::endl;
		return 1;
	}

	if(vm.count("config")){
		ifstream ifs(exec.configFile);
		if (!ifs)
		{
			cout << "can not open config file: " << exec.configFile << "\n";
			return 0;
		}
		else
		{
			po::options_description cConfigFile;
			cConfigFile.add(cLoader);
			cConfigFile.add(cExec);
			cConfigFile.add(cGrid);
			store(parse_config_file(ifs, cConfigFile), vm);
			notify(vm);
			ifs.close();
		}
	} else {
		cout << "No config file specified!" <<std::endl;
		return 1;
	}

	//define verbosity level
	exec.verbosity = Logger::cNORMAL;
	if(vm.count("silent"))
			exec.verbosity = Logger::cSILENT;
	if(vm.count("verbose"))
			exec.verbosity = Logger::cVERBOSE;
	if(vm.count("prolix"))
				exec.verbosity = Logger::cPROLIX;
	if(vm.count("live"))
					exec.verbosity = Logger::cLIVE;


	typedef std::pair<ExecutionParameters, EventDataLoadingParameters> tExecution;
	std::vector<tExecution> executions;

	//****************************************
	if(vm.count("testSuite")){

		ifstream ifs(testSuiteFile);
		if (!ifs)
		{
			cout << "can not open testSuite file: " << testSuiteFile << "\n";
			return 0;
		}
		else
		{
			std::vector<uint> threads;
			std::vector<uint> events;
			std::vector<uint> tracks;

			po::options_description cTestSuite;
			cTestSuite.add_options()
				("threads", po::value<std::vector<uint> >(&threads)->multitoken(), "work-group size")
				("tracks", po::value<std::vector<uint> >(&tracks)->multitoken(), "tracks to load")
				("eventGrouping", po::value<std::vector<uint> >(&events)->multitoken(), "events to process concurrently")
			;

			po::variables_map tests;
			po::store(parse_config_file(ifs, cTestSuite), tests);
			po::notify(tests);
			ifs.close();

			//add default values if necessary
			if(events.size() == 0)
				events.push_back(exec.eventGrouping);
			if(threads.size() == 0)
				threads.push_back(exec.threads);
			if(tracks.size() == 0)
				tracks.push_back(loader.maxTracks);

			//add test cases
			for(uint e : events){
				for(uint t : threads){
					for(uint n : tracks){

						ExecutionParameters ep(exec); //clone default
						EventDataLoadingParameters lp(loader);

						ep.threads = t;
						ep.eventGrouping = e; lp.maxEvents = e;
						lp.maxTracks = n;

						executions.push_back(std::make_pair(ep, lp));

					}
				}
			}
		}
	} else {
		executions.push_back(std::make_pair(exec, loader));
	}
	//************************************

	//**********************************
	RuntimeRecords runtimeRecords;
	for(uint e = 0; e < executions.size();++e){ //standard case: only 1
		std::cout << "Experiment " << e << ": ";
		for(uint i = 0; i < exec.iterations; ++i){
			std::cout << i+1 << "  " << std::flush;
			RuntimeRecords res = buildTriplets(executions[e].first, executions[e].second, grid);

			runtimeRecords.merge(res);
		}
		std::cout << std::endl;
	}
	//**********************************

	runtimeRecords.logPrint();

	std::stringstream runtimeOutputFile;
	runtimeOutputFile << "runtime" << (testSuiteFile != "" ? "." : "") << testSuiteFile << (exec.useCPU ? ".cpu" : ".gpu") << ".csv";

	std::ofstream runtimeRecordsFile(runtimeOutputFile.str(), std::ios::trunc);
	runtimeRecordsFile << runtimeRecords.csvDump();
	runtimeRecordsFile.close();

	std::cout << Logger::getInstance();

}
