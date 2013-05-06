cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(datastructures
	Hit.cpp
	HitCollection.cpp
	DetectorGeometry.cpp
	LayerSupplement.cpp
	EventSupplement.cpp
	serialize/Event.pb.cc
)

# needed by protobuf library to perform locking for 
# thread-safeness. The library is not using threads 
# internally however
# protobuffer version => 2.4 does not need this any more 
target_link_libraries(datastructures "pthread")

target_link_libraries(datastructures "protobuf")