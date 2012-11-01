#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include <clever/clever.hpp>

#include "../TrackletCollection.h"



TEST( TrackletCollection, test_create )
{
	TrackletCollection ht(1);

	const unsigned int id1 = 23;
	const unsigned int id2 = 0;
	const unsigned int id3 = 323;

	const unsigned int tk_id3 = 523;

	auto it = ht.getIterator();

	it.setValue<TrackletHit1>(id1);
	it.setValue<TrackletHit2>(id2);
	it.setValue<TrackletHit3>(id3);
	it.setValue<TrackletId>(tk_id3);

	GTEST_ASSERT_EQ( it.getValue<TrackletHit1>(),(id1));
	GTEST_ASSERT_EQ( it.getValue<TrackletHit2>(),(id2));
	GTEST_ASSERT_EQ( it.getValue<TrackletHit3>(),(id3));
	GTEST_ASSERT_EQ( it.getValue<TrackletId>(),(tk_id3));
}



