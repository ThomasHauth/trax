cmake_minimum_required (VERSION 2.6)
project (trax)

add_executable(HitCollection_stripdown
	HitCollection_stripdown.cpp
)
	
target_link_libraries ( HitCollection_stripdown datastructures )
	