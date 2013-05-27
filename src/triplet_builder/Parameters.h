#pragma once

#include <iostream>

class ExecutionParameters{

public:
	uint threads;
	uint eventGrouping;
	bool useCPU;
	uint iterations;
	int verbosity;
	std::string layerTripletConfigFile;
	std::string configFile;
	int layerTriplets;

};

class EventDataLoadingParameters{

public:
	std::string eventDataFile;
	int maxEvents;
	uint maxLayer;
	int maxTracks;
	float minPt;
	bool onlyTracks;
	std::string eventLoader;
};

std::ostream& operator<< (std::ostream& stream, const ExecutionParameters & exec);
std::ostream& operator<< (std::ostream& stream, const EventDataLoadingParameters & loader);
