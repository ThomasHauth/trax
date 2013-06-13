cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(algorithms
	PrefixSum.cpp
	GridBuilder.cpp
	PairGeneratorSector.cpp
	PairGeneratorBeamspot.cpp
	TripletThetaPhiPredictor.cpp
	TripletThetaPhiFilter.cpp
	TripletConnectivity.cpp
)