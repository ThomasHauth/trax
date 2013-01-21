#pragma once

class RuntimeRecord{

public:
	uint nTracks;

	float efficiency, fakeRate;

	float dataTransferWrite, dataTransferRead;
	float pairGenComp, pairGenStore;
	float tripletPredictComp, tripletPredictStore;
	float tripletCheckComp, tripletCheckStore;

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

};
