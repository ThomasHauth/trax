/*
 * LayerSupplement.cpp
 *
 *  Created on: Jan 30, 2013
 *      Author: dfunke
 */

#include <datastructures/LayerSupplement.h>

LayerInfo LayerSupplement::operator[](uint i){
	return LayerInfo(*this, i);
}

const LayerInfo LayerSupplement::operator[](uint i) const {
	return LayerInfo(*this, i);
}

