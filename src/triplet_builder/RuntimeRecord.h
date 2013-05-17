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

#include <boost/noncopyable.hpp>



using namespace clever;

struct tRuntimeInfo {
	ulong count;
	ulong scan;
	ulong store;
	ulong walltime;

	tRuntimeInfo() : count(0), scan(0), store(0), walltime(0){ }

	ulong totalKernel() const { return count + scan + store; }

	void startWalltime();
	void stopWalltime();

	tRuntimeInfo operator+(const tRuntimeInfo & rhs) const;

	std::string prettyPrint() const;
};

struct tIOInfo {
	ulong time;
	ulong bytes;

	float bandwith() const {
		return ((float) bytes) / time;
	}

	tIOInfo() : time(0), bytes(0) { }

	tIOInfo operator+(const tIOInfo & rhs) const;

	std::string prettyPrint() const;
};

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

	//RuntimeRecord operator+(const RuntimeRecord& rhs) const;

	//void operator+=(const RuntimeRecord& rhs);

};

class RuntimeRecordClass : private boost::noncopyable {

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

	RuntimeRecordClass(uint events_, uint layers_, uint layerTriplets_,
			uint hits_,  uint tracks_, uint threads_) {
		events = events_;
		layers = layers_;
		layerTriplets = layerTriplets_;

		hits = hits_;
		tracks = tracks_;

		threads = threads_;
	}

	void addRecord(RuntimeRecord r);

	const std::vector<RuntimeRecord> & getRecords() {
		return records;
	}

private:
	std::vector<RuntimeRecord> records;

};


