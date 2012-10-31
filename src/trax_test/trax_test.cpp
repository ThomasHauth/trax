
#include "gtest/gtest.h"


//#define USE_PERFTOOLS

#ifdef USE_PERFTOOLS
#include <google/profiler.h>
#endif


// include all tests here

// nice, we can easily include all the clever tests here to see
// if something breaks on the basic levels
#include <clever/test/clever_tests.h>

#include "../datastructures/test/DatastructuresTests.h"


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

#ifdef USE_PERFTOOLS
	ProfilerStart( "trax_test.prof");
#endif

	auto res =  RUN_ALL_TESTS();

#ifdef USE_PERFTOOLS
	ProfilerStop();
#endif

	return res;

}
