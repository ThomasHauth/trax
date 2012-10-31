#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>


TEST( DummyTracklets, test_create )
{
	clever::context contx;
	HitCollection ht;

	HitCollectionData::generatHitTestData( ht );

	HitCollectionTransfer clTrans;

	clTrans.initBuffers( contx, ht );

	clTrans.toDevice( contx, ht );

	// run kernel


	clTrans.fromDevice( contx, ht );
}
