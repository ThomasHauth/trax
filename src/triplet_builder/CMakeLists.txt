cmake_minimum_required (VERSION 2.6)
project (trax)

add_executable(tripletBuilder
	lib/ccolor.cpp
	RuntimeRecord.cpp
	Parameters.cpp
	TripletBuilder.cpp
)

target_link_libraries(tripletBuilder datastructures)
target_link_libraries(tripletBuilder algorithms)

target_link_libraries(tripletBuilder "OpenCL")
target_link_libraries(tripletBuilder "boost_program_options")
target_link_libraries(tripletBuilder "rt")

GET_TARGET_PROPERTY(TRIPLET_BUILDER_BIN tripletBuilder LOCATION)
ADD_CUSTOM_COMMAND(TARGET tripletBuilder
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{TRAX_DIR}/bin
          COMMAND ${CMAKE_COMMAND} -E copy ${TRIPLET_BUILDER_BIN} $ENV{TRAX_DIR}/bin/)