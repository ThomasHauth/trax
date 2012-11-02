cmake_minimum_required (VERSION 2.6)
project (trax)

# to check template reduction of various compilers
add_executable(HitCollection_stripdown
	HitCollection_stripdown.cpp
)
	
target_link_libraries ( HitCollection_stripdown datastructures )
target_link_libraries ( HitCollection_stripdown algorithms )

target_link_libraries(HitCollection_stripdown "OpenCL")



# main application to run OpenCL kernels of CMS input data
add_executable(trax_cms
	trax_cms.cpp
	EventProcessor.cpp
)
	
# internal libraries
target_link_libraries ( trax_cms datastructures )
target_link_libraries ( trax_cms algorithms )

# external libraries
target_link_libraries(trax_cms "OpenCL")
target_link_libraries(trax_cms "boost_program_options")
