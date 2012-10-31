cmake_minimum_required (VERSION 2.6)
project (trax)

# don't use pthreads within gtest. for simplicity and to not interfere with
# other threading libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0" )

add_executable(trax_test
 	${GTEST_BASE}/src/gtest-all.cc
	trax_test.cpp
)