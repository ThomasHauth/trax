cmake_minimum_required (VERSION 2.6)

project (trax)

include ( CMake_CompilerDefaults.txt )


message ("Setting up folder ${trax_SOURCE_DIR}")

# Include Exterals
set (EXTERNAL_BASE ${trax_SOURCE_DIR}/external/) 

# gtest
set (GTEST_BASE ${EXTERNAL_BASE}/gtest) 
include_directories (${GTEST_BASE}/include) 
include_directories (${GTEST_BASE}) 

# clever
include_directories (${EXTERNAL_BASE}/clever/src) 

#include_directories (${trax_SOURCE_DIR}/src/struct) 
#include_directories (${trax_SOURCE_DIR}/src/service) 
#include_directories (${trax_SOURCE_DIR}/src/eval) 
#include_directories (${trax_SOURCE_DIR}/src/algo) 

#boost
FIND_PACKAGE(Boost)

include_directories (${trax_SOURCE_DIR}) 

include_directories( ${INCLUDE_DIRECTORIES})

# add new header files here
set ( TRAX_HEADER 
"src/tech/test/openclam_test.h" 
)

message ( "header files in build " ${TRAX_HEADER}  ) 

set (TRAX_SRC src/algo/MPMultScatteringUpdater.cpp src/algo/TBBMultScatteringUpdater.cpp) 

#add_executable(trax_test src/eval/trax_test.cpp ${GTEST_BASE}/src/gtest-all.cc ${TRAX_SRC} )
#add_executable(trax_seeding src/eval/trax_seeding.cpp ${GTEST_BASE}/src/gtest-all.cc ${TRAX_SRC} )


add_subdirectory( algorithms )
add_subdirectory( datastructures )
add_subdirectory( trax_test )
add_subdirectory( trax_cms )
add_subdirectory( triplet_builder )


# ROOT libs ( not used, because we use the math lib in external )
#target_link_libraries(trax_eval ${ROOT_LIBRARIES})
#target_link_libraries(trax_test ${ROOT_LIBRARIES})

# open cl libs
#target_link_libraries(trax_test "OpenCL")
#target_link_libraries(trax_seeding "OpenCL")

# TBB libs
#target_link_libraries(trax_seeding "tbb")
#target_link_libraries(trax_test "tbb")


# google perf stuff ( not used right now )
#target_link_libraries(trax_test "profiler")
#target_link_libraries(trax_test "tcmalloc")

message ( "Include folders " ${CMAKE_INCLUDE_PATH}  ) 




