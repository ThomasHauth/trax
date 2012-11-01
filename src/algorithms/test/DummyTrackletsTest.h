#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>

#include <algorithms/DummyTracklets.h>


TEST( DummyTracklets, test_create )
{
	clever::context contx;
	HitCollection ht;

	HitCollectionData::generatHitTestData( ht, 30 );

	HitCollectionTransfer clTrans;

	clTrans.initBuffers( contx, ht );

	clTrans.toDevice( contx, ht );

	// run kernel
	DummyTracklets tkKernel ( contx );
	tkKernel.run( clTrans );

	clTrans.fromDevice( contx, ht );


}
