#include "RuntimeRecord.h"

float nsToMs(float ns) {
	return ns * 1E-6;
}

ulong sToNs(ulong ns) {
	return ns * 1E9;
}

#define PER * 1/

tRuntimeInfo tRuntimeInfo::operator+(const tRuntimeInfo & rhs) const {
	tRuntimeInfo result(*this);

	result.count += rhs.count;
	result.scan  += rhs.scan;
	result.store += rhs.store;
	result.walltime += rhs.walltime;

	return result;
}

std::string tRuntimeInfo::prettyPrint() const {
	std::stringstream s;

	s << "wall time: " << nsToMs(walltime) << " ms"
			<< " kernel time: " << nsToMs(totalKernel()) << " ms = ( "
			<< nsToMs(count) << " + " << nsToMs(scan) << "+" << nsToMs(store)
			<< " ) ms [Count + Scan + Store]";

	return s.str();
}

void tRuntimeInfo::startWalltime(){
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	walltime = sToNs(t.tv_sec)  + t.tv_nsec;
}

void tRuntimeInfo::stopWalltime(){
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	walltime = (sToNs(t.tv_sec) + t.tv_nsec) - walltime;
}

tIOInfo tIOInfo::operator+(const tIOInfo & rhs) const {
	tIOInfo result(*this);

	result.time += rhs.time;
	result.bytes += rhs.bytes;

	return result;
}

std::string tIOInfo::prettyPrint() const {

	std::stringstream s;

	s << bytes << " bytes in " << nsToMs(time) << " ms (" << bandwith() << " GB/s)";

	return s.str();

}

void fillInfo(tKernelEvent t, tRuntimeInfo & info){
	if(t.kernelName.find("Count") != std::string::npos){
		info.count = t.time;
	}
	if(t.kernelName.find("Store") != std::string::npos){
		info.store = t.time;
	}
	if(t.kernelName.find("prefixSum") != std::string::npos){
		info.scan += t.time;
	}
}

void RuntimeRecord::fillRuntimes(const clever::context & ctx) {

	//read
	tIOEvent read_ = ctx.getReadPerf();

	read.time = read_.time;
	read.bytes = read_.bytes;

	//write
	tIOEvent write_ = ctx.getWritePerf();

	write.time = write_.time;
	write.bytes = write_.bytes;

	//grid building
	for(cl_event e : GridBuilder::events){
		tKernelEvent t = ctx.getKernelPerf(e);
		fillInfo(t, buildGrid);
	}

	//pair gen
	for(cl_event e : PairGeneratorSector::events){
		tKernelEvent t = ctx.getKernelPerf(e);
		fillInfo(t, pairGen);
	}

	//triplet prediction
	for(cl_event e : TripletThetaPhiPredictor::events){
		tKernelEvent t = ctx.getKernelPerf(e);
		fillInfo(t, tripletPredict);
	}

	//triplet filter
	for(cl_event e : TripletThetaPhiFilter::events){
		tKernelEvent t = ctx.getKernelPerf(e);
		fillInfo(t, tripletFilter);
	}

}

void RuntimeRecord::logPrint() const {

	LOG << "Events: " << events << " Layers: " << layers << " LayerTriplets: " << layerTriplets
		<< " Hits: " << hits << " Loaded tracks: " << tracks <<std::endl;

	tIOInfo totalIO = write + read;
	LOG << "Total IO: " << totalIO.prettyPrint() << std::endl;
	VLOG << "Read: " << read.prettyPrint() << std::endl;
	VLOG << "Write: " << write.prettyPrint() << std::endl;

	tRuntimeInfo totalRuntime = buildGrid + pairGen + tripletPredict + tripletFilter;
	LOG << "Total runtime: ";
	LOG << totalRuntime.prettyPrint() << std::endl;
	LOG << "Wall time -- per event: " << nsToMs(totalRuntime.walltime PER events) << " ms"
		<< " -- per layerTriplet: " << nsToMs(totalRuntime.walltime PER (events*layerTriplets)) << " ms"
		<< " -- per track: " << nsToMs(totalRuntime.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Grid building: ";
	VLOG << buildGrid.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << nsToMs(buildGrid.walltime PER events) << " ms"
		<< " -- per layer " << nsToMs(buildGrid.walltime PER (events*layers)) << " ms"
		<< " -- per hit: " << nsToMs(buildGrid.walltime PER hits) << " ms" << std::endl;

	VLOG << std::endl << "Pair generation: ";
	VLOG << pairGen.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << nsToMs(pairGen.walltime PER events) << " ms"
		 << " -- per layerTriplet " << nsToMs(pairGen.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << nsToMs(pairGen.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet prediction: ";
	VLOG << tripletPredict.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << nsToMs(tripletPredict.walltime PER events) << " ms"
		 << " -- per layerTriplet " << nsToMs(tripletPredict.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << nsToMs(tripletPredict.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet filtering: ";
	VLOG << tripletFilter.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << nsToMs(tripletFilter.walltime PER events) << " ms"
		 << " -- per layerTriplet " << nsToMs(tripletFilter.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << nsToMs(tripletFilter.walltime PER tracks) << " ms" << std::endl;


}

/*RuntimeRecord RuntimeRecord::operator+(const RuntimeRecord& rhs) const {
	RuntimeRecord result(*this);

	result.nTracks += rhs.nTracks;
	result.dataTransferRead += rhs.dataTransferRead;
	result.dataTransferWrite += rhs.dataTransferWrite;
	result.pairGenComp += rhs.pairGenComp;
	result.pairGenStore += rhs.pairGenStore;
	result.tripletPredictComp += rhs.tripletPredictComp;
	result.tripletPredictStore += rhs.tripletPredictStore;
	result.tripletCheckComp += rhs.tripletCheckComp;
	result.tripletCheckStore += rhs.tripletCheckStore;
	result.buildGrid += rhs.buildGrid;

	result.efficiency = (this->nTracks*this->efficiency + rhs.nTracks*rhs.efficiency) / result.nTracks;
	result.fakeRate = (this->nTracks*this->fakeRate + rhs.nTracks*rhs.fakeRate) / result.nTracks;

	return result;
}

void RuntimeRecord::operator+=(const RuntimeRecord& rhs) {
	this->dataTransferRead += rhs.dataTransferRead;
	this->dataTransferWrite += rhs.dataTransferWrite;
	this->pairGenComp += rhs.pairGenComp;
	this->pairGenStore += rhs.pairGenStore;
	this->tripletPredictComp += rhs.tripletPredictComp;
	this->tripletPredictStore += rhs.tripletPredictStore;
	this->tripletCheckComp += rhs.tripletCheckComp;
	this->tripletCheckStore += rhs.tripletCheckStore;
	this->buildGrid += rhs.buildGrid;

	this->efficiency = (this->nTracks*this->efficiency + rhs.nTracks*rhs.efficiency) / (this->nTracks + rhs.nTracks);
	this->fakeRate = (this->nTracks*this->fakeRate + rhs.nTracks*rhs.fakeRate) / (this->nTracks + rhs.nTracks);

	this->nTracks += rhs.nTracks;
}*/

