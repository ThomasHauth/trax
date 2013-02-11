/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <iomanip>
#include <set>

#include <boost/program_options.hpp>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Dictionary.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>

#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/HitSorterZ.h>
#include <algorithms/HitSorterPhi.h>
#include <algorithms/PrefixSum.h>
#include <algorithms/BoundarySelectionZ.h>
#include <algorithms/BoundarySelectionPhi.h>

#include "RuntimeRecord.h"

#include "lib/ccolor.h"
#include "lib/CSV.h"

RuntimeRecord buildTriplets(uint tracks, float minPt, uint threads, bool verbose = false, bool useCPU = false) {
	//
	clever::context *contx;

	if(!useCPU){
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
	} else {
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

	//configure hit loader
	const uint maxLayer = 3;
	const uint nSectorsZ = 10;
	const uint nSectorsPhi = 8;
	LayerSupplement layerSupplement(maxLayer);
	Grid grid(maxLayer, nSectorsZ,nSectorsPhi);

	HitCollection hits;
	HitCollection::tTrackList validTracks = HitCollectionData::loadHitDataFromPB(hits, "hitsPXB_1k_SingleMu.pb", geom, layerSupplement, minPt, tracks,true, maxLayer);

	std::cout << "Loaded " << validTracks.size() << " tracks with minPt " << minPt << " GeV and " << hits.size() << " hits" << std::endl;

	if(verbose){
		for(uint i = 1; i <= maxLayer; ++i)
			std::cout << "Layer " << i << ": " << layerSupplement[i-1].getNHits() << " hits" << "\t Offset: " << layerSupplement[i-1].getOffset() << std::endl;


		//output sector borders
		/*std::cout << "Layer";
		for(int i = 0; i <= nSectorsPhi; ++i){
			std::cout << "\t" << -M_PI + i * (2*M_PI / nSectorsPhi);
		}
		std::cout << std::endl;

		for(int i = 0; i < maxLayer; i++){
			std::cout << i+1;
			for(int j = 0; j <= nSectorsPhi; j++){
				std::cout << "\t" << layerSupplement[i].sectorBorders[j];
			}
			std::cout << std::endl;
		}*/
	}

	/*****


	cl_device_id device;
	ERROR_HANDLER(
			ERROR = clGetCommandQueueInfo(contx->default_queue(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL));

	cl_ulong localMemSize;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL));
	cl_ulong maxAlloc;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxAlloc, NULL));
	size_t maxParam;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &maxParam, NULL));

	std::cout << "LocalMemSize: " << localMemSize << std::endl;
	std::cout << "MaxAlloc: " << maxAlloc << std::endl;
	std::cout << "MaxParam: " << maxParam << std::endl;



	 ********/

	//transer everything to gpu
	hits.transfer.initBuffers(*contx, hits);
	hits.transfer.toDevice(*contx, hits);

	geom.transfer.initBuffers(*contx, geom);
	geom.transfer.toDevice(*contx, geom);

	geomSupplement.transfer.initBuffers(*contx, geomSupplement);
	geomSupplement.transfer.toDevice(*contx, geomSupplement);

	dict.transfer.initBuffers(*contx, dict);
	dict.transfer.toDevice(*contx, dict);

	//transferring layer supplement
	layerSupplement.transfer.initBuffers(*contx, layerSupplement);
	layerSupplement.transfer.toDevice(*contx,layerSupplement);
	//initializating grid
	grid.transfer.initBuffers(*contx,grid);
	grid.config.upload(*contx);

	/*for(int i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << " Coordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		std::cout << " DetId: " << hit.getValue<DetectorId>() << " DetLayer: " << hit.getValue<DetectorLayer>();
		std::cout << " Event: " << hit.getValue<EventNumber>() << " HitId: " << hit.getValue<HitId>() << std::endl;
	}*/

	cl_ulong runtimeBuildGrid = 0;

	//sort hits on device in Z
	HitSorterZ sorterZ(*contx);
	runtimeBuildGrid += sorterZ.run(hits, threads,maxLayer,layerSupplement);

	//verify sorting
	bool valid = true;
	for(uint l = 1; l <= maxLayer; ++l){
		float lastZ = -9999;
		for(uint i = 0; i < layerSupplement[l-1].getNHits(); ++i){
			Hit hit(hits, layerSupplement[l-1].getOffset() + i);
			if(hit.globalZ() < lastZ){
				std::cerr << "Layer " << l << " : " << lastZ <<  "|" << hit.globalZ() << std::endl;
				valid = false;
			}
			lastZ = hit.globalZ();
		}
	}

	if(!valid)
		std::cerr << "Not sorted properly in Z" << std::endl;
	else
		std::cout << "Sorted correctly in Z" << std::endl;

	BoundarySelectionZ boundSelectZ(*contx);
	runtimeBuildGrid += boundSelectZ.run(hits, threads, maxLayer, layerSupplement, grid);

	//sort hits on device in Phi
	HitSorterPhi sorterPhi(*contx);
	runtimeBuildGrid += sorterPhi.run(hits, threads,maxLayer,layerSupplement, grid);

	//verify sorting
	valid = true;
	for(uint l = 1; l <= maxLayer; ++l){
		LayerGrid layerGrid(grid, l);
		for(uint s = 1; s <= grid.config.nSectorsZ; ++s){

			float lastPhi = - M_PI;
			for(uint h = layerGrid(s-1); h < layerGrid(s); ++h){
				Hit hit(hits, layerSupplement[l-1].getOffset() + h);

				//check phi
				if(hit.phi() < lastPhi){
					std::cerr << "Layer " << l << " : " << lastPhi <<  "|" << hit.phi() << std::endl;
					valid = false;
				}
				lastPhi = hit.phi();

				//check correct z sector
				if(!(grid.config.boundaryValuesZ[s-1] <= hit.globalZ() && hit.globalZ() <= grid.config.boundaryValuesZ[s])){
					std::cerr << "Layer " << l << " : zAct: " << hit.globalZ() <<  " in sector [" << grid.config.boundaryValuesZ[s-1] << ", " << grid.config.boundaryValuesZ[s] << "]" << std::endl;
					valid = false;
				}
			}
		}
	}

	if(!valid)
		std::cerr << "Not sorted properly in Phi" << std::endl;
	else
		std::cout << "Sorted correctly in Phi" << std::endl;



	/*for(int i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << " Coordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		std::cout << " Phi: " << atan2(hit.globalY(), hit.globalX()) << std::endl;
	}*/

	BoundarySelectionPhi boundSelectPhi(*contx);
	runtimeBuildGrid += boundSelectPhi.run(hits, threads, maxLayer, layerSupplement, grid);

	//output grid
	for(uint l = 1; l <= maxLayer; ++l){
		std::cout << "Layer: " << l << std::endl;

		//output z boundaries
		std::cout << "z/phi\t\t";
		for(uint i = 0; i <= grid.config.nSectorsZ; ++i){
			std::cout << grid.config.boundaryValuesZ[i] << "\t";
		}
		std::cout << std::endl;

		LayerGrid layerGrid(grid, l);
		for(uint p = 0; p <= grid.config.nSectorsPhi; ++p){
			std::cout << std::setprecision(3) << grid.config.boundaryValuesPhi[p] << "\t\t";
			for(uint z = 0; z <= grid.config.nSectorsZ; ++z){
				std::cout << layerGrid(z,p) << "\t";
			}
			std::cout << std::endl;
		}
	}



	for(uint i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << "\tTrack: " << hit.getValue<HitId>();
		std::cout << " \tCoordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		//std::cout << " DetId: " << hit.getValue<DetectorId>() << " DetLayer: " << hit.getValue<DetectorLayer>();
		//std::cout << " Event: " << hit.getValue<EventNumber>() << " HitId: " << hit.getValue<HitId>();
		std::cout << std::endl;
	}

	//prefix sum test
	/*std::vector<uint> uints(19,100);
	uints.push_back(0);
	clever::vector<uint,1> dUints(uints, *contx);

	PrefixSum psum(*contx);
	uint res = psum.run(dUints, uints.size(), 4, true);
	transfer::download(dUints, uints, *contx);

	for(uint i = 0; i < uints.size(); ++i){
		std::cout << i << ":" << uints[i] << "\t";
	}

	std::cout << std::endl << "Result: " << res << std::endl;

	return RuntimeRecord();*/

	// configure kernel

	int layers[] = {1,2,3};

	float dThetaCut = 0.05;
	float dThetaWindow = 0.1;
	float dPhiCut = 0.1;
	float dPhiWindow = 0.1;
	int pairSpreadZ = 1;

	//run it
	TripletThetaPhiFilter tripletThetaPhi(*contx);
	TrackletCollection * tracklets = tripletThetaPhi.run(hits, geom, geomSupplement, dict, threads, layers,
			layerSupplement, grid, dThetaCut, dThetaWindow, dPhiCut, dPhiWindow, pairSpreadZ);

	//evaluate it
	std::set<uint> foundTracks;
	uint fakeTracks = 0;

	std::cout << "Found " << tracklets->size() << " triplets:" << std::endl;
	for(uint i = 0; i < tracklets->size(); ++i){
		Tracklet tracklet(*tracklets, i);

		if(tracklet.isValid(hits)){
			//valid triplet
			foundTracks.insert(tracklet.trackId(hits));
			if(verbose){
				std::cout << zkr::cc::fore::green;
				std::cout << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
				std::cout << zkr::cc::console << std::endl;
			}
		}
		else {
			//fake triplet
			++fakeTracks;
			if(verbose){
				std::cout << zkr::cc::fore::red;
				std::cout << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
				std::cout << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
				std::cout << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
				std::cout << zkr::cc::console << std::endl;
			}
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
	result.buildGrid = runtimeBuildGrid;
	std::cout << "Build Grid: " << runtimeBuildGrid << "ns" << std::endl;

	profile_info pinfo = contx->report_profile(contx->PROFILE_WRITE);
	result.dataTransferWrite = pinfo.runtime();
	std::cout << "Data Transfer\tWritten: " << pinfo.runtime() << "ns\tRead: ";
	pinfo = contx->report_profile(contx->PROFILE_READ);
	result.dataTransferRead = pinfo.runtime();
	std::cout << pinfo.runtime() << " ns" << std::endl;


	pinfo = contx->report_profile(PairGeneratorSector::KERNEL_COMPUTE_EVT());
	std::cout << "Pair Generation\tCompute: " << pinfo.runtime() << " ns\tStore: ";
	result.pairGenComp = pinfo.runtime();
	pinfo = contx->report_profile(PairGeneratorSector::KERNEL_STORE_EVT());
	result.pairGenStore = pinfo.runtime();
	std::cout << pinfo.runtime() << " ns" << std::endl;


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
	bool verbose;
	bool useCPU;

	po::options_description desc("Allowed Options");
	desc.add_options()
			("help", "produce help message")
			("minPt", po::value<float>(&minPt)->default_value(1.0), "minimum track Pt")
			("tracks", po::value<uint>(&tracks)->default_value(100), "number of valid tracks to load")
			("threads", po::value<uint>(&threads)->default_value(256), "number of threads to use")
			("silent", po::value<bool>(&silent)->zero_tokens(), "supress all messages from TripletFinder")
			("verbose", po::value<bool>(&verbose)->zero_tokens(), "elaborate information")
			("use-cpu", po::value<bool>(&useCPU)->zero_tokens(), "force using CPU instead of GPU")
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

		results << "#nTracks, dataTransfer, buildGrid, pairGen, tripletPredict, tripletFilter, computation, runtime, efficiency, fakeRate" << std::endl;

		for(uint i : testCases){
			RuntimeRecord res = buildTriplets(i,minPt, threads, useCPU);

			results << res.nTracks << ", " << res.totalDataTransfer() << ", " << res.buildGrid << ", " << res.totalPairGen() << ", " << res.totalTripletPredict() << ", " << res.totalTripletCheck() << ", "  << res.totalComputation() << ", " << res.totalRuntime()
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

	RuntimeRecord res = buildTriplets(tracks,minPt, threads, verbose, useCPU);

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
