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

int main(int argc, char *argv[]) {

	//global variables
	HitCollection ht;
	clever::context contx;
	HitCollectionTransfer clTrans_hits;

	const int maxLayer = 3;
	int hitCount[maxLayer] = { }; // all elements 0

	const int tracks = 10;
	const double minPt = 1;

	HitCollection::tTrackList validTracks = HitCollectionData::loadHitDataFromPB(ht, "/home/dfunke/devel/trax/build/triplet_builder/hitsPXB.pb", hitCount, minPt, tracks,true, maxLayer);

	std::cout << "Loaded " << tracks << " tracks with minPt " << minPt << " GeV and " << ht.size() << " hits" << std::endl;
	for(int i = 1; i <= maxLayer; ++i)
		std::cout << "Layer " << i << ": " << hitCount[i-1] << " hits" << std::endl;


	clTrans_hits.initBuffers(contx, ht);
	clTrans_hits.toDevice(contx, ht);

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
		std::cout << "hit1: " << tracklet.hit1() << " [" << ht.getValue(HitId(),tracklet.hit1()) <<"]";
		std::cout << " hit2: " << tracklet.hit2() << " [" << ht.getValue(HitId(),tracklet.hit2()) <<"]";
		std::cout << " hit3: " << tracklet.hit3() << " [" << ht.getValue(HitId(),tracklet.hit3()) <<"]" << std::endl;

		if(ht.getValue(HitId(),tracklet.hit1()) == ht.getValue(HitId(),tracklet.hit2())
				&& ht.getValue(HitId(),tracklet.hit1()) == ht.getValue(HitId(),tracklet.hit3()))
			//valid triplet
			foundTracks.insert(ht.getValue(HitId(),tracklet.hit1()));
		else
			//fake triplet
			++fakeTracks;
	}

	std::cout << "Efficiency: " << ((double) foundTracks.size()) / validTracks.size() << " FakeRate: " << ((double) fakeTracks) / tracklets->size() << std::endl;

	//output not found tracks
	for(auto vTrack : validTracks) {
		if( foundTracks.find(vTrack.first) == foundTracks.end())
			std::cout << "Didn't find track " << vTrack.first << std::endl;
	}
}
