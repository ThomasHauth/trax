#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

#include <algorithms/TripletConnectivity.h>

class TripletConnectivityTestSupport
{
public:
	TripletConnectivityTestSupport()
	{

	}

	void addSimpleForward()
	{
		// id, hit1id, hit2id, hit3id, connectivity
		trBase.addWithValue(0, 1, 2, 3, 0);
		trBase.addWithValue(1, 1, 2, 4, 0);

		trFollowing.addWithValue(2, 3, 5, 6, 0);
		trFollowing.addWithValue(3, 9, 4, 8, 0);
	}

	void addSimpleBackward()
	{
		// id, hit1id, hit2id, hit3id, connectivity
		trBase.addWithValue(0, 3, 2, 23, 0);
		trBase.addWithValue(1, 5, 2, 1, 0);

		trFollowing.addWithValue(2, 12, 8, 3, 0);
		trFollowing.addWithValue(3, 15, 11, 8, 0);
	}

	void initAndTransfer()
	{
		trBaseTrans.initBuffers(contx, trBase);
		trBaseTrans.toDevice(contx, trBase);

		trFollowingTrans.initBuffers(contx, trFollowing);
		trFollowingTrans.toDevice(contx, trFollowing);
	}

	TrackletCollection trBase;
	TrackletCollection trFollowing;

	TrackletCollectionTransfer trBaseTrans;
	TrackletCollectionTransfer trFollowingTrans;

	clever::context contx;
};

TEST( TripletConnectivity, test_run )
{
	TripletConnectivityTestSupport testSupport;
	testSupport.addSimpleForward();
	testSupport.initAndTransfer();

	TripletConnectivity ttc(testSupport.contx);

	ttc.run(testSupport.trBaseTrans, testSupport.trFollowingTrans, false);
	testSupport.contx.finish_default_queue();

	testSupport.trFollowingTrans.fromDevice(testSupport.contx,
			testSupport.trFollowing);

	// check if the connectivity has been increased correctly
	auto it = testSupport.trFollowing.getIterator();
	GTEST_ASSERT_EQ( 1, it.getValue<TrackletConnectivity>( ));
	it++;
	GTEST_ASSERT_EQ( 0, it.getValue<TrackletConnectivity>( ));
}

TEST( TripletConnectivity, test_run_backward )
{
	TripletConnectivityTestSupport testSupport;
	testSupport.addSimpleBackward();
	testSupport.initAndTransfer();

	TripletConnectivity ttc(testSupport.contx);

	ttc.run(testSupport.trBaseTrans, testSupport.trFollowingTrans, true);
	testSupport.contx.finish_default_queue();

	testSupport.trFollowingTrans.fromDevice(testSupport.contx,
			testSupport.trFollowing);

	// check if the connectivity has been increased correctly
	auto it = testSupport.trFollowing.getIterator();
	GTEST_ASSERT_EQ( 1, it.getValue<TrackletConnectivity>( ));
	it++;
	GTEST_ASSERT_EQ( 0, it.getValue<TrackletConnectivity>( ));
}


//global variables
/*HitCollection ht;
 clever::context contx;
 TrackletCollection* tracklets;
 HitCollectionTransfer clTrans_hits;
 TrackletCollectionTransfer clTrans_tracklet;*/
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
