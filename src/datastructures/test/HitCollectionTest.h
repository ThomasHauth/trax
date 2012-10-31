#pragma once

#include <iostream>
#include <gtest/gtest.h>

#include "../HitCollection.h"

TEST( HitCollection, test_create ) {
	HitCollection ht;

	float fX = 23.0f;
	float fY = 5.0f;

	GlobalX gx;
	GlobalY gy;

	ht.setValue(gy, fY);
	ht.setValue(gx, fX);

	const float outx =  ht.getValue( gx );
	const float outy =  ht.getValue( gy );

	std::cout << outx << outy;

	GTEST_ASSERT_EQ( ht.getValue( gx ), fX);
	GTEST_ASSERT_EQ( ht.getValue( gy ), fY);
}
