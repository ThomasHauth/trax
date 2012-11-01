#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include "../HitCollection.h"



TEST( HitCollection, test_create )
{
	HitCollection ht(1);

	float fX = 23.0f;
	float fY = 5.0f;

	auto it = ht.getIterator();

	it.setValue<GlobalX>(fX);
	it.setValue<GlobalY>(fY);

	GTEST_ASSERT_EQ( it.getValue<GlobalX>( ), fX);
	GTEST_ASSERT_EQ( it.getValue<GlobalY>( ), fY);
}

TEST( HitCollection, addItem )
{
	HitCollection ht(1);

	ht.addEntry();

	GTEST_ASSERT_EQ( ht.size(), (unsigned int) 2);
}

TEST( HitCollection, addItemWithValue )
{
	HitCollection ht;

	float fX = 23.0f;
	float fY = 5.0f;

	ht.addWithValue(fX, fY, 0.0, 0, 0);

	auto it = ht.getIterator();

	GTEST_ASSERT_EQ( it.getValue<GlobalX>( ), fX);
	GTEST_ASSERT_EQ( it.getValue<GlobalY>( ), fY);
}

TEST( HitCollection, hitclass )
{
	HitCollection ht;

	// this will create a new entry
	Hit h(ht);

	GTEST_ASSERT_EQ( ht.size(), (unsigned int)1);
}

TEST( HitCollection, OpenCLTransfer )
{
	HitCollection ht;
	clever::context contx;

	float fX = 23.0f;
	float fY = 5.0f;

	ht.addWithValue( fX, fY, 0.0, 0,0 );

	HitCollectionTransfer clTrans;

	clTrans.initBuffers( contx, ht );

	clTrans.toDevice( contx, ht );
	clTrans.fromDevice( contx, ht );

	auto it = ht.getIterator();

	GTEST_ASSERT_EQ( fX, it.getValue<GlobalX>( ));
	GTEST_ASSERT_EQ( fY, it.getValue<GlobalY>( ));
}

TEST( HitCollection, OpenCLTransfer_large )
{
	HitCollection ht( 400000 );
	clever::context contx;

	HitCollectionTransfer clTrans;

	clTrans.initBuffers( contx, ht );

	clTrans.toDevice( contx, ht );
	clTrans.fromDevice( contx, ht );
}


