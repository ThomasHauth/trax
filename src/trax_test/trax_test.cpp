
#include "gtest/gtest.h"


//#define USE_PERFTOOLS

#ifdef USE_PERFTOOLS
#include <google/profiler.h>
#endif


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
