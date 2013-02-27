#pragma once

class RuntimeRecord{

public:
	uint nTracks;

	float efficiency, fakeRate;

	float dataTransferWrite, dataTransferRead;
	float pairGenComp, pairGenStore;
	float tripletPredictComp, tripletPredictStore;
	float tripletCheckComp, tripletCheckStore;
	float buildGrid;

	RuntimeRecord() {
		dataTransferRead =0;
		dataTransferWrite =0;
		pairGenComp =0;
		pairGenStore =0;
		tripletPredictComp =0;
		tripletPredictStore =0;
		tripletCheckComp =0;
		tripletCheckStore =0;
		buildGrid =0;

		efficiency = 0;
		fakeRate = 0;

		nTracks =0;
	}

	float totalDataTransfer() const {
		return dataTransferRead+dataTransferWrite;
	}

	float totalPairGen() const {
		return pairGenComp + pairGenStore;
	}

	float totalTripletPredict() const {
		return tripletPredictComp + tripletPredictStore;
	}

	float totalTripletCheck() const {
		return tripletCheckComp + tripletCheckStore;
	}

	float totalComputation() const {
		return totalPairGen() + totalTripletPredict() + totalTripletCheck();
	}

	float totalRuntime() const {
		return totalComputation() + totalDataTransfer();
	}

	RuntimeRecord operator+(const RuntimeRecord& rhs) const {
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

	void operator+=(const RuntimeRecord& rhs) {
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
	}

};
