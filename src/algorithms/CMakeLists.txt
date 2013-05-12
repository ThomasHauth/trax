cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(algorithms
	TripletConnectivity.cpp
	PrefixSum.cpp
	GridBuilder.cpp
	PairGeneratorSector.cpp
)