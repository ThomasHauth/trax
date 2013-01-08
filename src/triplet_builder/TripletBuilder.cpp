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

#include <algorithms/TripletThetaPhiFilter.h>

#include "lib/ccolor.cpp"

int main(int argc, char *argv[]) {

	//global variables
	HitCollection hits;
	clever::context contx;
	HitCollectionTransfer clTrans_hits;

	const int maxLayer = 3;
	int hitCount[maxLayer] = { }; // all elements 0

	const int tracks = 100;
	const double minPt = .3;

	HitCollection::tTrackList validTracks = HitCollectionData::loadHitDataFromPB(hits, "/home/dfunke/devel/trax/build/triplet_builder/hitsPXB.pb", hitCount, minPt, tracks,true, maxLayer);

	std::cout << "Loaded " << validTracks.size() << " tracks with minPt " << minPt << " GeV and " << hits.size() << " hits" << std::endl;
	for(int i = 1; i <= maxLayer; ++i)
		std::cout << "Layer " << i << ": " << hitCount[i-1] << " hits" << std::endl;


	clTrans_hits.initBuffers(contx, hits);
	clTrans_hits.toDevice(contx, hits);

	// run kernel
	TripletThetaPhiFilter tripletThetaPhi(contx);

	int layers[] = {1,2,3};

	float dTheta = 0.01;
	float dPhi = 0.1;

	TrackletCollection * tracklets = tripletThetaPhi.run(clTrans_hits, 4, layers, hitCount, dTheta, dPhi);

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
