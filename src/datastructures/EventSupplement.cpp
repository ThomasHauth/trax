/*
 * LayerSupplement.cpp
 *
 *  Created on: Jan 30, 2013
 *      Author: dfunke
 */

#include <datastructures/EventSupplement.h>

EventInfo EventSupplement::operator[](uint i){
	return EventInfo(*this, i);
}

const EventInfo EventSupplement::operator[](uint i) const {
	return EventInfo(*this, i);
}

