#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>

#include <datastructures/Logger.h>

class KernelWrapper : private boost::noncopyable {

public:


	KernelWrapper(clever::context & ctext) : ctx(ctext) { }

protected:

	clever::context & ctx;

	#define PRINTF(a)       printf a  // print debug output
	//#define PRINTF(a)                 // no debug output
	static const std::string oclDEFINES;

};
