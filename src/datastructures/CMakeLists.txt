cmake_minimum_required (VERSION 2.6)
project (trax)

add_library(datastructures
	Hit.cpp
	HitCollection.cpp
	serialize/Event.pb.cc
)

# needed by protobuf library to perform locking for 
# thread-safeness. The library is not using threads 
# internally however
target_link_libraries(datastructures "pthread")

target_link_libraries(datastructures "protobuf")