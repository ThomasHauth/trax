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

	void addFourLayerForward()
	{
		// triplets on four layers:
		//  -- outside --
		//trBaseTrans  			 Hits:  ( 1  - 10 )
		//trFollowingTrans       Hits:  ( 11 - 20 )
		//trFollowingTwoTrans    Hits:  ( 21 - 30 )
		//trFollowingThreeTrans  Hits:  ( 31 - 40 )
		//  -- inside ---

		unsigned int triptletId = 1;

		// id, hit1id, hit2id, hit3id, connectivity
		trBase.addWithValue(triptletId++, 1, 3, 8, 0);
		trBase.addWithValue(triptletId++, 1, 3, 5, 0);
		trBase.addWithValue(triptletId++, 2, 4, 6, 0);/* track */
		trBase.addWithValue(triptletId++, 2, 4, 7, 0);

		trFollowing.addWithValue(triptletId++, 6, 11, 16, 0);/* track */
		trFollowing.addWithValue(triptletId++, 9, 12, 17, 0);
		trFollowing.addWithValue(triptletId++, 9, 13, 18, 0);
		trFollowing.addWithValue(triptletId++, 9, 14, 19, 0);

		trFollowingTwo.addWithValue(triptletId++, 16, 21, 30, 0);/* track */
		trFollowingTwo.addWithValue(triptletId++, 15, 22, 29, 0);
		trFollowingTwo.addWithValue(triptletId++, 15, 24, 29, 0);
		trFollowingTwo.addWithValue(triptletId++, 15, 25, 27, 0);

		trFollowingThree.addWithValue(triptletId++, 30, 31, 36, 0);/*track*/
		trFollowingThree.addWithValue(triptletId++, 28, 32, 37, 0);
		trFollowingThree.addWithValue(triptletId++, 26, 33, 38, 0);
		trFollowingThree.addWithValue(triptletId++, 27, 34, 39, 0);
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

		if (trFollowing.size() > 0)
		{
			trFollowingTrans.initBuffers(contx, trFollowing);
			trFollowingTrans.toDevice(contx, trFollowing);
		}

		if (trFollowingTwo.size() > 0)
		{
			trFollowingTwoTrans.initBuffers(contx, trFollowingTwo);
			trFollowingTwoTrans.toDevice(contx, trFollowingTwo);
		}

		if (trFollowingThree.size() > 0)
		{
			trFollowingThreeTrans.initBuffers(contx, trFollowingThree);
			trFollowingThreeTrans.toDevice(contx, trFollowingThree);
		}
	}

	TrackletCollection trBase;
	TrackletCollection trFollowing;
	TrackletCollection trFollowingTwo;
	TrackletCollection trFollowingThree;

	TrackletCollectionTransfer trBaseTrans;
	TrackletCollectionTransfer trFollowingTrans;
	TrackletCollectionTransfer trFollowingTwoTrans;
	TrackletCollectionTransfer trFollowingThreeTrans;

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

TEST( TripletConnectivity, test_run_multilayer )
{
	TripletConnectivityTestSupport testSupport;
	testSupport.addFourLayerForward();
	testSupport.initAndTransfer();

	TripletConnectivity ttc(testSupport.contx);

	ttc.run(testSupport.trBaseTrans, testSupport.trFollowingTrans, false);
	ttc.run(testSupport.trFollowingTrans, testSupport.trFollowingTwoTrans,
			false);
	ttc.run(testSupport.trFollowingTwoTrans, testSupport.trFollowingThreeTrans,
			false);

	testSupport.contx.finish_default_queue();

	// download all
	testSupport.trFollowingTrans.fromDevice(testSupport.contx,
			testSupport.trFollowing);
	testSupport.trFollowingTwoTrans.fromDevice(testSupport.contx,
			testSupport.trFollowingTwo);
	testSupport.trFollowingThreeTrans.fromDevice(testSupport.contx,
			testSupport.trFollowingThree);

	// check if the connectivity has been increased correctly
	// the first one should be the tracklet which is part of a track
	auto it = testSupport.trFollowingThree.getIterator();
	GTEST_ASSERT_EQ( 3, it.getValue<TrackletConnectivity>( ));
	it++;
	GTEST_ASSERT_EQ( 0, it.getValue<TrackletConnectivity>( ));
	it++;
	GTEST_ASSERT_EQ( 0, it.getValue<TrackletConnectivity>( ));
	it++;
	GTEST_ASSERT_EQ( 1, it.getValue<TrackletConnectivity>( ));
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
