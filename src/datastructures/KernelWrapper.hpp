#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>

#include <datastructures/Logger.h>

#include <vector>

//Curiously recurring template pattern (CRTP)
template <typename T>
class KernelWrapper : private boost::noncopyable {

public:


	KernelWrapper(clever::context & ctext) : ctx(ctext) { }

protected:

	clever::context & ctx;

	#define PRINTF(a)       printf a  // print debug output
	//#define PRINTF(a)                 // no debug output
	static const std::string oclDEFINES;

public:
	static std::vector<cl_event> events;

};

template <typename T> const std::string KernelWrapper<T>::oclDEFINES = 	 PROLIX ? "#define PRINTF(a) printf a" :
														  	  	  	  	  	  	  	  "#define PRINTF(a)";

template <typename T> std::vector<cl_event> KernelWrapper<T>::events;
