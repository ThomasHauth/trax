#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include "../HitCollection.h"

TEST( HitCollection, test_create ) {
	HitCollection ht(1);

	float fX = 23.0f;
	float fY = 5.0f;

	auto it = ht.getIterator();

	it.setValue<GlobalX>( fX);
	it.setValue<GlobalY>( fY);

	GTEST_ASSERT_EQ( it.getValue<GlobalX>( ), fX);
	GTEST_ASSERT_EQ( it.getValue<GlobalY>( ), fY);
}

TEST( HitCollection, addItem ) {
	HitCollection ht(1);

	ht.addItem();

	GTEST_ASSERT_EQ( ht.size(), 2);
}
