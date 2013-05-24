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
#include <boost/filesystem.hpp>

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
#include "PhysicsRecord.h"
#include "Parameters.h"
#include "EventLoader.h"

#include "lib/ccolor.h"
#include "lib/CSV.h"


std::string traxDir = getenv("TRAX_DIR");

clever::context * createContext(ExecutionParameters exec){

	//
	LOG << "Creating context for " << (exec.useCPU ? "CPU" : "GPGPU") << "...";

//#ifndef CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL
#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL 0
//#endif

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

	return contx;
}

std::pair<RuntimeRecords, PhysicsRecords> buildTriplets(ExecutionParameters exec, EventDataLoadingParameters loader, GridConfig gridConfig, clever::context * contx) {

	RuntimeRecords runtimeRecords;
	PhysicsRecords physicsRecords;

	{ //block to ensure proper destruction

		//load radius dictionary
		Dictionary dict;

		std::ifstream radiusDictFile(traxDir + "/configs/radiusDictionary.dat");
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

		std::ifstream detectorGeometryFile(traxDir + "/configs/detectorRadius.dat");
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
		loader.maxLayer = layerConfig.loadTripletConfigurationFromFile(traxDir + "/configs/" + exec.layerTripletConfigFile, 1); //TODO one defined for testing

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
				PB_Event::PEvent pEvent = edLoader->getEvent();

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
					PhysicsRecord physics(e, p);
					physics.fillData(*tracklets, validTracks[e], hits, layerConfig.size());
					physicsRecords.addRecord(physics);
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

		delete edLoader;

	} //destruction order block

	//delete contx;

	return std::make_pair(runtimeRecords, physicsRecords);
}

std::string getFilename (const std::string& str)
{
	//std::cout << "Splitting: " << str << '\n';
	unsigned found = str.find_last_of("/\\");
	//std::cout << " path: " << str.substr(0,found) << '\n';
	//std::cout << " file: " << str.substr(found+1) << '\n';

	return str.substr(found+1);
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
		ifstream ifs(traxDir + "/configs/" + getFilename(exec.configFile));
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

		ifstream ifs(traxDir + "/configs/" + getFilename(testSuiteFile));
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

	//set up logger verbosity
	Logger::getInstance().setLogLevel(exec.verbosity);
	clever::context * contx = createContext(exec);

	RuntimeRecords runtimeRecords;
	PhysicsRecords physicsRecords;
	for(uint e = 0; e < executions.size();++e){ //standard case: only 1
		std::cout << "Experiment " << e << ": " << std::endl;
		std::cout << executions[e].first << " " << executions[e].second << std::endl;
		for(uint i = 0; i < exec.iterations; ++i){
			std::cout << i+1 << "  " << std::flush;
			auto res = buildTriplets(executions[e].first, executions[e].second, grid, contx);

			contx->clearAllBuffers();

			runtimeRecords.merge(res.first);
			physicsRecords.merge(res.second);
		}
		std::cout << std::endl;
	}

	delete contx;
	//**********************************

	runtimeRecords.logPrint();

	std::stringstream outputFileRuntime;
	outputFileRuntime << traxDir << "/runtime/" << "runtime" << (testSuiteFile != "" ? "." : "") << getFilename(testSuiteFile) << (exec.useCPU ? ".cpu" : ".gpu") << ".csv";

	std::ofstream runtimeRecordsFile(outputFileRuntime.str(), std::ios::trunc);
	runtimeRecordsFile << runtimeRecords.csvDump();
	runtimeRecordsFile.close();

	std::stringstream outputFilePhysics;
	outputFilePhysics << traxDir << "/physics/" << "physics." << getFilename(exec.configFile) << ".csv";
	std::stringstream outputDirPhysics;
	outputDirPhysics << traxDir << "/physics/" << getFilename(exec.configFile);

	boost::filesystem::create_directories(outputDirPhysics.str());

	std::ofstream physicsRecordsFile(outputFilePhysics.str(), std::ios::trunc);
	physicsRecordsFile << physicsRecords.csvDump(outputDirPhysics.str());
	physicsRecordsFile.close();

	std::cout << Logger::getInstance();

}
