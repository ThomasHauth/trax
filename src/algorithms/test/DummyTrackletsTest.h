#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

#include <algorithms/DummyTracklets.h>

TEST( DummyTracklets, test_create )
{
	clever::context contx;
	HitCollection ht;

	const unsigned int hitCount = 30;
	const unsigned int maxTrackletCount = (hitCount - 2) * (hitCount - 1)
			* hitCount;

	// must be large enough to hold all combinations
	TrackletCollection tracklets(maxTrackletCount);

	HitCollectionData::generatHitTestData(ht, 30);

	HitCollectionTransfer clTrans;
	TrackletCollectionTransfer clTrans_tracklet;

	clTrans.initBuffers(contx, ht);
	clTrans.toDevice(contx, ht);

	clTrans_tracklet.initBuffers(contx, tracklets);

	// run kernel
	DummyTracklets tkKernel(contx);
	unsigned int found = tkKernel.run(clTrans, clTrans_tracklet);

	clTrans_tracklet.fromDevice(contx, tracklets);

	ASSERT_EQ( maxTrackletCount, found);
}
