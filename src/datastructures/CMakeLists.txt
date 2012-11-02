cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(datastructures
	Hit.cpp
	HitCollection.cpp
	serialize/Event.pb.cc
)


target_link_libraries(datastructures "protobuf")