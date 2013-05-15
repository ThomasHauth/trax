#pragma once

class ExecutionParameters{

public:
	uint threads;
	uint eventGrouping;
	bool useCPU;
	int verbosity;
	std::string layerTripletConfigFile;
	std::string configFile;

};

class EventDataLoadingParameters{

public:
	std::string eventDataFile;
	int maxEvents;
	uint maxLayer;
	int maxTracks;
	float minPt;
	bool onlyTracks;
	bool singleEventLoader;
};
