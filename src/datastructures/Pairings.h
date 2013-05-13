#pragma once

#include <boost/noncopyable.hpp>
#include <clever/clever.hpp>

using namespace clever;
using namespace std;

class Pairing : private boost::noncopyable {

private:
	clever::context & ctx;
	mutable std::vector<uint2> * lPairing;
	mutable std::vector<uint> * lPairingOffsets;

public:
	clever::vector<uint2,1> pairing;
	clever::vector<uint, 1> pairingOffsets;

public:

	Pairing(clever::context & ctext, const uint nTotalPairings, const uint nEvents, const uint nLayerTriplets)
		: ctx(ctext), lPairing(NULL), lPairingOffsets(NULL),
		  pairing(ctx, nTotalPairings),
		  pairingOffsets(0 , nEvents*nLayerTriplets+1, ctx) {	}

	const std::vector<uint> & getPairingOffsets() const {
		if(lPairingOffsets != NULL)
			return *lPairingOffsets;
		else {
			lPairingOffsets = new std::vector<uint>(pairingOffsets.get_count());
			transfer::download(pairingOffsets, *lPairingOffsets, ctx);

			return *lPairingOffsets;
		}
	}

	const std::vector<uint2> & getPairings() const {
		if(lPairing != NULL)
			return *lPairing;
		else {
			lPairing = new std::vector<uint2>(pairing.get_count());
			transfer::download(pairing, *lPairing, ctx);

			return *lPairing;
		}
	}

	void invalidate(){
		delete lPairing; lPairing = NULL;
		delete lPairingOffsets; lPairingOffsets = NULL;
	}

	~Pairing(){
		delete lPairing;
		delete lPairingOffsets;
	}

};
