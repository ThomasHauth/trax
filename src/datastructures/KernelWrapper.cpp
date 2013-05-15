/*
 * KernelWrapper.cpp
 *
 *  Created on: May 15, 2013
 *      Author: dfunke
 */

#include "KernelWrapper.h"

const std::string KernelWrapper::oclDEFINES = 	 PROLIX ? "#define PRINTF(a) printf a" :
														  "#define PRINTF(a)";
