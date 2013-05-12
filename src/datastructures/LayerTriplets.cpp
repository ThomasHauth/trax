/*
 * LayerSupplement.cpp
 *
 *  Created on: Jan 30, 2013
 *      Author: dfunke
 */

#include "LayerTriplets.h"

LayerTriplet LayerTriplets::operator[](uint i){
	return LayerTriplet(*this, i);
}

const LayerTriplet LayerTriplets::operator[](uint i) const {
	return LayerTriplet(*this, i);
}

