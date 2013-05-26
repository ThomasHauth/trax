/*
 * Parameters.cpp
 *
 *  Created on: May 21, 2013
 *      Author: dfunke
 */

#include "Parameters.h"

std::ostream& operator<< (std::ostream& stream, const ExecutionParameters & exec){
	stream << "Threads: " << exec.threads << " Evt groups " << exec.eventGrouping << " Layer triplets: " << exec.layerTriplets;

	return stream;
}

std::ostream& operator<< (std::ostream& stream, const EventDataLoadingParameters & loader){
	stream << "Max evts: " << loader.maxEvents << " Max tracks: " << loader.maxTracks;

	return stream;
}


