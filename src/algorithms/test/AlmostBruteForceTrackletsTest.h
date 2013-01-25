#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

#include <algorithms/AlmostBruteForceTracklets.h>

//global variables
HitCollection ht;
clever::context contx;
TrackletCollection* tracklets;
HitCollectionTransfer clTrans_hits;
TrackletCollectionTransfer clTrans_tracklet;
/*
TEST( AlmostBruteForceTracklets, test_create )
{

	HitCollectionData::loadHitDataFromPB(ht, "/home/dfunke/devel/trax/build/trax_test/hitsPXB_TIB_TOB.pb", 0, 10,false, 3);

	std::cout << "Loaded " << ht.size() << " hits" << std::endl;

	const unsigned int hitCount = ht.size();
	const unsigned int maxTrackletCount = (hitCount - 2) * (hitCount - 1)
					* hitCount;

	// must be large enough to hold all combinations
	tracklets = new TrackletCollection(maxTrackletCount);

	std::cout << "Reserving space for " << maxTrackletCount << " tracklets" << std::endl;
}

TEST( AlmostBruteForceTracklets, OpenCL_Transfer)
{
	clTrans_hits.initBuffers(contx, ht);
	clTrans_hits.toDevice(contx, ht);
	clTrans_tracklet.initBuffers(contx, *tracklets);

}

TEST( AlmostBruteForceTracklets, run_kernel)
{
	// run kernel
	AlmostBruteForceTracklets tkKernel(contx);
	tkKernel.run(clTrans_hits, clTrans_tracklet);
}

TEST( AlmostBruteForceTracklets, OpenCL_Result_Transfer)
{
	clTrans_tracklet.fromDevice(contx, *tracklets);
}
*/