cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(algorithms
	PrefixSum.cpp
	GridBuilder.cpp
	PairGeneratorSector.cpp
	TripletThetaPhiPredictor.cpp
	TripletThetaPhiConstrainedPredictor.cpp
	TripletThetaPhiFilter.cpp
	TripletConnectivity.cpp
)