#pragma once

#include <clever/clever.hpp>

#include <sstream>
#include <time.h>
#include <vector>

#include <datastructures/Logger.h>

#include <algorithms/GridBuilder.h>
#include <algorithms/PairGeneratorSector.h>
#include <algorithms/TripletThetaPhiPredictor.h>
#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/PrefixSum.h>



using namespace clever;

struct tRuntimeInfo {
	long double count;
	long double scan;
	long double store;
	long double walltime;

	tRuntimeInfo() : count(0), scan(0), store(0), walltime(0){ }

	long double totalKernel() const { return count + scan + store; }

	void startWalltime();
	void stopWalltime();

	tRuntimeInfo operator+(const tRuntimeInfo & rhs) const;

	std::string prettyPrint() const;
	std::string prettyPrint(const tRuntimeInfo & var) const;

	std::string csvDump() const;
	std::string csvDump(const tRuntimeInfo & var) const;
};

struct tIOInfo {
	long double time;
	long double bytes;

	float bandwith() const {
		return ((float) bytes) / time;
	}

	tIOInfo() : time(0), bytes(0) { }

	tIOInfo operator+(const tIOInfo & rhs) const;

	std::string prettyPrint() const;
	std::string prettyPrint(const tIOInfo & var) const;

	std::string csvDump() const;
	std::string csvDump(const tIOInfo & var) const;
};

class RuntimeRecordClass;

class RuntimeRecord{

public:
	uint events;
	uint layers;
	uint layerTriplets;

	uint threads;

	uint hits;
	uint tracks;

	tIOInfo read;
	tIOInfo write;

	tRuntimeInfo buildGrid, pairGen, tripletPredict, tripletFilter;

	RuntimeRecord(uint events_, uint layers_, uint layerTriplets_,
					uint hits_,  uint tracks_, uint threads_) {
		events = events_;
		layers = layers_;
		layerTriplets = layerTriplets_;

		hits = hits_;
		tracks = tracks_;

		threads = threads_;
	}


	void fillRuntimes(const clever::context & ctx);

	void logPrint() const;
	std::string csvDump() const;

	bool operator==(const RuntimeRecord & r) const;
	bool operator==(const RuntimeRecordClass & r) const;

	//RuntimeRecord operator+(const RuntimeRecord& rhs) const;

	//void operator+=(const RuntimeRecord& rhs);

};

class RuntimeRecordClass {

public:
	uint events;
	uint layers;
	uint layerTriplets;

	uint threads;

	uint hits;
	uint tracks;

	tIOInfo readMean, readVar;
	tIOInfo writeMean, writeVar;

	tRuntimeInfo buildGridMean, buildGridVar;
	tRuntimeInfo pairGenMean, pairGenVar;
	tRuntimeInfo tripletPredictMean, tripletPredictVar;
	tRuntimeInfo tripletFilterMean, tripletFilterVar;

	RuntimeRecordClass(uint events_, uint layers_, uint layerTriplets_,
			uint hits_,  uint tracks_, uint threads_) {
		events = events_;
		layers = layers_;
		layerTriplets = layerTriplets_;

		hits = hits_;
		tracks = tracks_;

		threads = threads_;
	}

	RuntimeRecordClass(const RuntimeRecord & r){
		events = r.events;
		layers = r.layers;
		layerTriplets = r.layerTriplets;

		hits = r.hits;
		tracks = r.tracks;

		threads = r.threads;
	}

	tRuntimeInfo toVar(tRuntimeInfo m2) const;
	tIOInfo toVar(tIOInfo m2) const;

	void addRecord(const RuntimeRecord & r);
	void merge(const RuntimeRecordClass & c);

	const std::vector<RuntimeRecord> & getRecords() {
		return records;
	}

	void logPrint() const;
	std::string csvDump() const;

	bool operator==(const RuntimeRecord & r) const;
	bool operator==(const RuntimeRecordClass & r) const;

private:
	std::vector<RuntimeRecord> records;

};

class RuntimeRecords {

private:
	std::vector<RuntimeRecordClass> classes;

public:
	void addRecord(const RuntimeRecord & r);
	void addRecordClass(const RuntimeRecordClass & c);
	void merge(const RuntimeRecords & c);

	void logPrint() const;
	std::string csvDump() const;

};


