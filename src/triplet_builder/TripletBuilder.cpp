/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <set>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.cpp>
#include <datastructures/Dictionary.h>

#include <algorithms/TripletThetaPhiFilter.h>

#include "lib/ccolor.cpp"
#include "lib/CSV.h"

int main(int argc, char *argv[]) {

	//
	clever::context *contx;
	try{
		//try gpu
		contx = new clever::context(clever::context_settings::default_gpu());
		std::cout << "Using GPGPU" << std::endl;
	} catch (const std::runtime_error & e){
		//if not use cpu
		contx = new clever::context();
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
	int hitCount[maxLayer] = { }; // all elements 0

	const int tracks = 100;
	const double minPt = 1;

	HitCollection hits;
	HitCollection::tTrackList validTracks = HitCollectionData::loadHitDataFromPB(hits, "hitsPXB.pb", geom, hitCount, minPt, tracks,true, maxLayer);

	std::cout << "Loaded " << validTracks.size() << " tracks with minPt " << minPt << " GeV and " << hits.size() << " hits" << std::endl;
	for(int i = 1; i <= maxLayer; ++i)
		std::cout << "Layer " << i << ": " << hitCount[i-1] << " hits" << std::endl;


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
	TrackletCollection * tracklets = tripletThetaPhi.run(hitTransfer, geomTransfer, dictTransfer, 4, layers, hitCount, dTheta, dPhi);

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

	//output not found tracks
	for(auto vTrack : validTracks) {
		if( foundTracks.find(vTrack.first) == foundTracks.end())
			std::cout << "Didn't find track " << vTrack.first << std::endl;
	}
}
