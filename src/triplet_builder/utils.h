/*
 * utils.h
 *
 *  Created on: May 24, 2013
 *      Author: dfunke
 */

#pragma once

#include <sstream>

#define PER * 1/
#define SEP ' '

class Utils{
public:

	static long double nsToMs(long double ns) {
		return ns * 1E-6;
	}

	static long double sToNs(long double ns) {
		return ns * 1E9;
	}

	static std::string csv(std::initializer_list<long double> args){
		std::stringstream s;

		for(auto it = args.begin(); it != args.end(); ++it){
			s << *it;
			if((it + 1) != args.end())
				s << SEP;
		}

		return s.str();

	}

	static std::string csv(std::initializer_list<uint> args){
		std::stringstream s;

		for(auto it = args.begin(); it != args.end(); ++it){
			s << *it;
			if((it + 1) != args.end())
				s << SEP;
		}

		return s.str();

	}

	static std::string csv(std::initializer_list<std::string> args){
		std::stringstream s;

		for(auto it = args.begin(); it != args.end(); ++it){
			s << *it;
			if((it + 1) != args.end())
				s << SEP;
		}

		return s.str();

	}

	static uint clamp(int n){
		return n > 0 ? n : 1;
	}

};
