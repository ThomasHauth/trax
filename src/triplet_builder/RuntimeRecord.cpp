#include "RuntimeRecord.h"


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

	s << "wall time: " << Utils::nsToMs(walltime) << " ms"
			<< " kernel time: " << Utils::nsToMs(totalKernel()) << " ms = ( "
			<< Utils::nsToMs(count) << " + " << Utils::nsToMs(scan) << "+" << Utils::nsToMs(store)
			<< " ) ms [Count + Scan + Store]";

	return s.str();
}

std::string tRuntimeInfo::prettyPrint(const tRuntimeInfo & var) const {
	std::stringstream s;

	s << "wall time: " << Utils::nsToMs(walltime) << " +/- " << Utils::nsToMs(var.walltime) << " ms"
			<< " kernel time: " << Utils::nsToMs(totalKernel()) << " +/- " << Utils::nsToMs(var.totalKernel()) << " ms = ( "
			<< Utils::nsToMs(count) << " +/- " << Utils::nsToMs(var.count) << " + " << Utils::nsToMs(scan) << " +/- " << Utils::nsToMs(var.scan) << "+" << Utils::nsToMs(store) << " +/- " << Utils::nsToMs(var.store)
			<< " ) ms [Count + Scan + Store]";

	return s.str();
}

std::string tRuntimeInfo::csvDump() const {
	std::stringstream s;

	s << Utils::csv({ Utils::nsToMs(count), Utils::nsToMs(scan), Utils::nsToMs(store), Utils::nsToMs(walltime), Utils::nsToMs(totalKernel()) });

	return s.str();
}

std::string tRuntimeInfo::csvDump(const tRuntimeInfo & var) const{
	std::stringstream s;

	s << Utils::csv({ Utils::nsToMs(count), Utils::nsToMs(var.count), Utils::nsToMs(scan), Utils::nsToMs(var.scan), Utils::nsToMs(store), Utils::nsToMs(var.store), Utils::nsToMs(walltime), Utils::nsToMs(var.walltime), Utils::nsToMs(totalKernel()), Utils::nsToMs(var.totalKernel()) });

	return s.str();
}

void tRuntimeInfo::startWalltime(){
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	walltime = Utils::sToNs(t.tv_sec)  + t.tv_nsec;
}

void tRuntimeInfo::stopWalltime(){
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	walltime = (Utils::sToNs(t.tv_sec) + t.tv_nsec) - walltime;
}

tIOInfo tIOInfo::operator+(const tIOInfo & rhs) const {
	tIOInfo result(*this);

	result.time += rhs.time;
	result.bytes += rhs.bytes;

	return result;
}

std::string tIOInfo::prettyPrint() const {

	std::stringstream s;

	s << bytes << " bytes in " << Utils::nsToMs(time) << " ms (" << bandwith() << " GB/s)";

	return s.str();

}

std::string tIOInfo::prettyPrint(const tIOInfo & var) const {

	std::stringstream s;

	s << bytes << " bytes in " << Utils::nsToMs(time) << " +/- " << Utils::nsToMs(var.time) << " ms (" << bandwith() << " GB/s)";

	return s.str();

}

std::string tIOInfo::csvDump() const {
	std::stringstream s;

	s << Utils::csv({ Utils::nsToMs(time), bytes });

	return s.str();
}

std::string tIOInfo::csvDump(const tIOInfo & var) const{
	std::stringstream s;

	s << Utils::csv({ Utils::nsToMs(time), Utils::nsToMs(var.time), bytes, var.bytes });

	return s.str();
}

void fillInfo(tKernelEvent t, tRuntimeInfo & info){
	if(t.kernelName.find("Count") != std::string::npos){
		info.count += t.time;
	}
	if(t.kernelName.find("Store") != std::string::npos){
		info.store += t.time;
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
	LOG << "Wall time -- per event: " << Utils::nsToMs(totalRuntime.walltime PER events) << " ms"
		<< " -- per layerTriplet: " << Utils::nsToMs(totalRuntime.walltime PER (events*layerTriplets)) << " ms"
		<< " -- per track: " << Utils::nsToMs(totalRuntime.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Grid building: ";
	VLOG << buildGrid.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(buildGrid.walltime PER events) << " ms"
		<< " -- per layer " << Utils::nsToMs(buildGrid.walltime PER (events*layers)) << " ms"
		<< " -- per hit: " << Utils::nsToMs(buildGrid.walltime PER hits) << " ms" << std::endl;

	VLOG << std::endl << "Pair generation: ";
	VLOG << pairGen.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(pairGen.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(pairGen.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(pairGen.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet prediction: ";
	VLOG << tripletPredict.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(tripletPredict.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(tripletPredict.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(tripletPredict.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet filtering: ";
	VLOG << tripletFilter.prettyPrint() << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(tripletFilter.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(tripletFilter.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(tripletFilter.walltime PER tracks) << " ms" << std::endl;


}

void RuntimeRecordClass::logPrint() const {

	LOG << "Events: " << events << " Layers: " << layers << " LayerTriplets: " << layerTriplets
		<< " Hits: " << hits << " Loaded tracks: " << tracks << std::endl;
	LOG << "Measurements: " << records.size() << std::endl;

	tIOInfo totalIOMean = writeMean + readMean;
	tIOInfo totalIOVar = writeVar + readVar;
	LOG << "Total IO: " << totalIOMean.prettyPrint(toVar(totalIOVar)) << std::endl;
	VLOG << "Read: " << readMean.prettyPrint(toVar(readVar)) << std::endl;
	VLOG << "Write: " << writeMean.prettyPrint(toVar(writeVar)) << std::endl;

	tRuntimeInfo totalRuntimeMean = buildGridMean + pairGenMean + tripletPredictMean + tripletFilterMean;
	tRuntimeInfo totalRuntimeVar = buildGridVar + pairGenVar + tripletPredictVar + tripletFilterVar;
	LOG << "Total runtime: ";
	LOG << totalRuntimeMean.prettyPrint(toVar(totalRuntimeVar)) << std::endl;
	LOG << "Wall time -- per event: " << Utils::nsToMs(totalRuntimeMean.walltime PER events) << " ms"
		<< " -- per layerTriplet: " << Utils::nsToMs(totalRuntimeMean.walltime PER (events*layerTriplets)) << " ms"
		<< " -- per track: " << Utils::nsToMs(totalRuntimeMean.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Grid building: ";
	VLOG << buildGridMean.prettyPrint(toVar(buildGridVar)) << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(buildGridMean.walltime PER events) << " ms"
		<< " -- per layer " << Utils::nsToMs(buildGridMean.walltime PER (events*layers)) << " ms"
		<< " -- per hit: " << Utils::nsToMs(buildGridMean.walltime PER hits) << " ms" << std::endl;

	VLOG << std::endl << "Pair generation: ";
	VLOG << pairGenMean.prettyPrint(toVar(pairGenVar)) << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(pairGenMean.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(pairGenMean.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(pairGenMean.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet prediction: ";
	VLOG << tripletPredictMean.prettyPrint(toVar(tripletPredictVar)) << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(tripletPredictMean.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(tripletPredictMean.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(tripletPredictMean.walltime PER tracks) << " ms" << std::endl;

	VLOG << std::endl << "Triplet filtering: ";
	VLOG << tripletFilterMean.prettyPrint(toVar(tripletFilterVar)) << std::endl;
	VLOG << "Wall time -- per event: " << Utils::nsToMs(tripletFilterMean.walltime PER events) << " ms"
		 << " -- per layerTriplet " << Utils::nsToMs(tripletFilterMean.walltime PER (events*layerTriplets)) << " ms"
		 << " -- per track: " << Utils::nsToMs(tripletFilterMean.walltime PER tracks) << " ms" << std::endl;


}

void RuntimeRecords::logPrint() const{
	LOG << "Experiments: " << classes.size() << std::endl;

	for(uint i = 0; i < classes.size(); ++i){
		classes[i].logPrint();
	}
}

bool RuntimeRecord::operator==(const RuntimeRecord & r) const{
	return	this->events == r.events
		 && this->layerTriplets == r.layerTriplets
	     && this->layers == r.layers
	     && this->threads == r.threads
	     && this->hits == r.hits
	     && this->tracks == r.tracks;
}

std::string RuntimeRecord::csvDump() const {
	std::stringstream s;

	tRuntimeInfo total = buildGrid + pairGen + tripletPredict + tripletFilter;

	s << Utils::csv({events, layers, layerTriplets, threads, hits, tracks}) << SEP; //header
	s << read.csvDump() << SEP << write.csvDump() << SEP; //IO
	s << buildGrid.csvDump() << SEP << pairGen.csvDump() << SEP << tripletPredict.csvDump() << SEP << tripletFilter.csvDump() << SEP << total.csvDump(); //runtime

	return s.str();
}

bool RuntimeRecord::operator==(const RuntimeRecordClass & r) const{
	return	this->events == r.events
		 && this->layerTriplets == r.layerTriplets
	     && this->layers == r.layers
	     && this->threads == r.threads
	     && this->hits == r.hits
	     && this->tracks == r.tracks;
}

bool RuntimeRecordClass::operator==(const RuntimeRecord & r) const{
	return	this->events == r.events
		 && this->layerTriplets == r.layerTriplets
	     && this->layers == r.layers
	     && this->threads == r.threads
	     && this->hits == r.hits
	     && this->tracks == r.tracks;
}

std::string RuntimeRecordClass::csvDump() const {
	std::stringstream s;

	tRuntimeInfo totalMean = buildGridMean + pairGenMean + tripletPredictMean + tripletFilterMean;
	tRuntimeInfo totalVar = buildGridVar + pairGenVar + tripletPredictVar + tripletFilterVar;

	s << Utils::csv({events, layers, layerTriplets, threads, hits, tracks, (uint) records.size()}) << SEP; //header
	s << readMean.csvDump(toVar(readVar)) << SEP << writeMean.csvDump(toVar(writeVar)) << SEP; //IO
	s << buildGridMean.csvDump(toVar(buildGridVar)) << SEP << pairGenMean.csvDump(toVar(pairGenVar))
	  << SEP << tripletPredictMean.csvDump(toVar(tripletPredictVar)) << SEP << tripletFilterMean.csvDump(toVar(tripletFilterVar)) << SEP << totalMean.csvDump(toVar(totalVar)); //runtime

	return s.str();
}

bool RuntimeRecordClass::operator==(const RuntimeRecordClass & r) const{
	return	this->events == r.events
		 && this->layerTriplets == r.layerTriplets
	     && this->layers == r.layers
	     && this->threads == r.threads
	     && this->hits == r.hits
	     && this->tracks == r.tracks;
}

tRuntimeInfo RuntimeRecordClass::toVar(tRuntimeInfo m2) const{
	tRuntimeInfo result;

	result.count = std::sqrt(m2.count / Utils::clamp(records.size() - 1));
	result.scan = std::sqrt(m2.scan / Utils::clamp(records.size() - 1));
	result.store = std::sqrt(m2.store / Utils::clamp(records.size() - 1));
	result.walltime = std::sqrt(m2.walltime / Utils::clamp(records.size() - 1));

	return result;
}

tIOInfo RuntimeRecordClass::toVar(tIOInfo m2) const{
	tIOInfo result;

	result.time = std::sqrt(m2.time / Utils::clamp(records.size() - 1));
	result.bytes = std::sqrt(m2.bytes / Utils::clamp(records.size() - 1));

	return result;
}

//calculate moving mean and variance
//n is INclusive new element
void calculateMeanVar(tRuntimeInfo & mean, tRuntimeInfo & var, const tRuntimeInfo & x, uint n){

	long double delta = x.count - mean.count;
	mean.count += delta / Utils::clamp(n);
	var.count +=  delta*(x.count - mean.count);

	delta = x.scan - mean.scan;
	mean.scan += delta / Utils::clamp(n);
	var.scan +=  delta*(x.scan - mean.scan);

	delta = x.store - mean.store;
	mean.store += delta / Utils::clamp(n);
	var.store +=  delta*(x.store - mean.store);;

	delta = x.walltime - mean.walltime;
	mean.walltime += delta / Utils::clamp(n);
	var.walltime +=  delta*(x.walltime - mean.walltime);

}

void calculateMeanVar(tIOInfo & mean, tIOInfo & var, const tIOInfo & x, uint n){

	long double delta = x.time - mean.time;
	mean.time += delta / Utils::clamp(n);
	var.time += delta*(x.time - mean.time);


	//bytes should actually be pretty much the same
	delta = x.bytes - mean.bytes;
	mean.bytes += delta / Utils::clamp(n);
	var.bytes += delta*(x.bytes - mean.bytes);

}



void RuntimeRecordClass::addRecord(const RuntimeRecord & r){

	if(*this == r){

		records.push_back(r);

		calculateMeanVar(buildGridMean, buildGridVar, r.buildGrid, records.size());
		calculateMeanVar(pairGenMean, pairGenVar, r.pairGen, records.size());
		calculateMeanVar(tripletPredictMean, tripletPredictVar, r.tripletPredict, records.size());
		calculateMeanVar(tripletFilterMean, tripletFilterVar, r.tripletFilter, records.size());

		calculateMeanVar(readMean, readVar, r.read, records.size());
		calculateMeanVar(writeMean, writeVar, r.write, records.size());

	}

}

void RuntimeRecordClass::merge(const RuntimeRecordClass & c){

	//transfer all values
	for(uint i = 0; i < c.records.size(); ++i){
		records.push_back(c.records[i]);
		addRecord(c.records[i]);
	}

}

void RuntimeRecords::addRecord(const RuntimeRecord & r){

	for(uint i = 0; i < classes.size(); ++i){

		if(classes[i] == r){
			classes[i].addRecord(r);

			return;
		}

	}

	//class not yet present, lets add it
	RuntimeRecordClass c(r);
	c.addRecord(r);
	classes.push_back(c);

}

void RuntimeRecords::addRecordClass(const RuntimeRecordClass & c){
	for(uint i = 0; i < classes.size(); ++i){

		if(classes[i] == c){
			classes[i].merge(c);
			return;
		}
	}

	classes.push_back(c);
}

void RuntimeRecords::merge(const RuntimeRecords & c){

	//add all record classes --> get merged if necessary
	for(uint i = 0; i < c.classes.size(); ++i){
		addRecordClass(c.classes[i]);
	}

}

std::string RuntimeRecords::csvDump() const {
	std::stringstream s;

	//header
	s << Utils::csv({"events", "layers", "layerTriplets", "threads", "hits", "tracks", "n"}) << SEP; //header
	s << Utils::csv({"readTime", "readTimeVar", "readBytes", "readBytesVar"}) << SEP; //read
	s << Utils::csv({"writeTime", "writeTimeVar", "writeBytes", "writeBytesVar"}) << SEP; //write
	s << Utils::csv({"buildGridCount", "buildGridCountVar", "buildGridScan", "buildGridScanVar", "buildGridStore", "buildGridStoreVar", "buildGridWalltime", "buildGridWalltimeVar", "buildGridKernel", "buildGridKernelVar"}) << SEP; //buildGrid
	s << Utils::csv({"pairGenCount", "pairGenCountVar", "pairGenScan", "pairGenScanVar", "pairGenStore", "pairGenStoreVar", "pairGenWalltime", "pairGenWalltimeVar", "pairGenKernel", "pairGenKernelVar"}) << SEP; //pairGen
	s << Utils::csv({"tripletPredictCount", "tripletPredictCountVar", "tripletPredictScan", "tripletPredictScanVar", "tripletPredictStore", "tripletPredictStoreVar", "tripletPredictWalltime", "tripletPredictWalltimeVar", "tripletPredictKernel", "tripletPredictKernelVar"}) << SEP; //tripletPredict
	s << Utils::csv({"tripletFilterCount", "tripletFilterCountVar", "tripletFilterScan", "tripletFilterScanVar", "tripletFilterStore", "tripletFilterStoreVar", "tripletFilterWalltime", "tripletFilterWalltimeVar", "tripletFilterKernel", "tripletFilterKernelVar"}) << SEP; //tripletFilter
	s << Utils::csv({"totalCount", "totalCountVar", "totalScan", "totalScanVar", "totalStore", "totalStoreVar", "totalWalltime", "totalWalltimeVar", "totalKernel", "totalKernelVar"}); //totalTiming
	s << std::endl;

	for(auto i : classes)
		s << i.csvDump() << std::endl;

	return s.str();
}
