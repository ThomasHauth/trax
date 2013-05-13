/*
 * TripletThetaPhiFilter.cpp
 *
 *  Created on: May 13, 2013
 *      Author: dfunke
 */

#include "TripletThetaPhiFilter.h"

#define DEBUG_OUT
TrackletCollection * TripletThetaPhiFilter::run(HitCollection & hits, const Grid & grid,
		const Pairing & pairs, const Pairing & tripletCandidates,
		int nThreads, const LayerTriplets & layerTriplets,
		float dThetaCut, float dPhiCut, float tipCut)
{
	std::vector<uint> oracleOffset;
	uint totalCandidates = 0;

	uint nLayerTriplets = layerTriplets.size();
	std::vector<uint> nFoundCandidates = tripletCandidates.getPairingOffsets();

	for(uint e = 0; e < grid.config.nEvents; ++e){
		for(uint p = 0; p < nLayerTriplets; ++p){

			//plus 1 for offset shift [0] == 0 so nFoundHits for [0] is [1]
			uint candidates = nFoundCandidates[e * nLayerTriplets + p + 1];
			candidates = 32 * std::ceil(candidates / 32.0); //round to next multiple of 32

			oracleOffset.push_back(totalCandidates);
			totalCandidates += candidates;
		}
	}

	std::cout << "Initializing oracle offsets for triplet prediction...";
	clever::vector<uint, 1> m_oracleOffset(oracleOffset, ctx);
	std::cout << "done[" << m_oracleOffset.get_count()  << "]" << std::endl;

	std::cout << "Initializing oracle...";
	clever::vector<uint, 1> m_oracle(0, std::ceil(totalCandidates / 32.0), ctx);
	std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

	std::cout << "Initializing prefix sum...";
	clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*nLayerTriplets*nThreads+1, ctx);
	std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

	std::cout << "Running filter kernel...";
	cl_event evt = tripletThetaPhiCheck.run(
			//configuration
			dThetaCut, dPhiCut, tipCut,
			// input
			pairs.pairing.get_mem(),
			tripletCandidates.pairing.get_mem(), tripletCandidates.pairingOffsets.get_mem(),
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			// output
			m_oracle.get_mem(),m_oracleOffset.get_mem(),  m_prefixSum.get_mem(),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	std::cout << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

#ifdef DEBUG_OUT
	std::cout << "Fetching prefix sum...";
	std::vector<uint> cPrefixSum(m_prefixSum.get_count());
	transfer::download(m_prefixSum,cPrefixSum,ctx);
	std::cout << "done" << std::endl;
	std::cout << "Prefix sum: ";
	for(auto i : cPrefixSum){
		std::cout << i << " ; ";
	}
	std::cout << std::endl;
#endif

#ifdef DEBUG_OUT
	std::cout << "Fetching oracle...";
	std::vector<uint> oracle(m_oracle.get_count());
	transfer::download(m_oracle,oracle,ctx);
	std::cout << "done" << std::endl;
	std::cout << "Oracle: ";
	for(auto i : oracle){
		std::cout << i << " ; ";
	}
	std::cout << std::endl;
#endif

	//Calculate prefix sum
	PrefixSum prefixSum(ctx);
	evt = prefixSum.run(m_prefixSum.get_mem(), m_prefixSum.get_count(), nThreads);
	uint nFoundTriplets;
	transfer::downloadScalar(m_prefixSum, nFoundTriplets, ctx, true, m_prefixSum.get_count()-1, 1, &evt);


#ifdef DEBUG_OUT
	std::cout << "Fetching prefix sum...";
	transfer::download(m_prefixSum,cPrefixSum,ctx);
	std::cout << "done" << std::endl;
	std::cout << "Prefix sum: ";
	for(auto i : cPrefixSum){
		std::cout << i << " ; ";
	}
	std::cout << std::endl;
#endif

	TrackletCollection * tracklets = new TrackletCollection(nFoundTriplets, grid.config.nEvents, layerTriplets.size(), ctx);
	std::cout << "Reserving space for " << nFoundTriplets << " tracklets" << std::endl;

	tracklets->transfer.initBuffers(ctx, *tracklets);

	std::cout << "Running filter store kernel...";
	evt = tripletThetaPhiStore.run(
			//input
			pairs.pairing.get_mem(),
			tripletCandidates.pairing.get_mem(), tripletCandidates.pairingOffsets.get_mem(),
			m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
			// output
			tracklets->transfer.buffer(TrackletHit1()), tracklets->transfer.buffer(TrackletHit2()), tracklets->transfer.buffer(TrackletHit3()),
			tracklets->trackletOffsets.get_mem(),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	std::cout << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_STORE_EVT());

	std::cout << "Fetching triplets...";
	tracklets->transfer.fromDevice(ctx, *tracklets);
	std::cout <<"done[" << tracklets->size() << "]" << std::endl;

#ifdef DEBUG_OUT
	std::cout << "Fetching triplet offets...";
	std::vector<uint> tripOffsets = tracklets->getTrackletOffsets();
	std::cout <<"done[" << tripOffsets.size() << "]" << std::endl;

	std::cout << "Tracklet Offsets:" << std::endl;
	for(uint i = 0; i < tripOffsets.size(); ++i){
		std::cout << "[" << i << "] "  << tripOffsets[i] << std::endl;
	}
#endif

	return tracklets;
}

clever::vector<uint2,1> * TripletThetaPhiFilter::generateAllPairs(HitCollection & hits, int nThreads, int layers[], const LayerSupplement & layerSupplement) {

	int nLayer1 = layerSupplement[layers[0]-1].getNHits();
	int nLayer2 = layerSupplement[layers[1]-1].getNHits();

	int nMaxPairs = nLayer1 * nLayer2;
	std::vector<uint2> pairs;
	for(int i = 0; i < nLayer1; ++i)
		for(int j=0; j < nLayer2; ++j)
			pairs.push_back(uint2(layerSupplement[layers[0]-1].getOffset() + i,layerSupplement[layers[1]-1].getOffset() + j));

	std::cout << "Transferring " << pairs.size() << " pairs...";
	clever::vector<uint2,1> * m_pairs = new clever::vector<uint2,1>(pairs, nMaxPairs, ctx);
	int nPairs = m_pairs->get_count();
	std::cout << "done[" << nPairs  << "]" << std::endl;

	return m_pairs;
}

clever::vector<uint2,1> * TripletThetaPhiFilter::generateAllTriplets(HitCollection & hits, int nThreads, int layers[], int hitCount[],
		float dThetaWindow, float dPhiWindow, const clever::vector<uint2,1> & pairs) {

	int nLayer1 = hitCount[layers[0]-1];
	int nLayer2 = hitCount[layers[1]-1];
	int nLayer3 = hitCount[layers[2]-1];

	int nPairs = pairs.get_count();
	int nTriplets = nPairs * nLayer3;

	std::vector<uint2> triplets;
	for(int i = 0; i < nPairs; ++i)
		for(int j = 0; j < nLayer3; ++j)
			triplets.push_back(uint2(i,nLayer1 + nLayer2 + j));

	std::cout << "Transferring " << triplets.size() << " triplets...";
	clever::vector<uint2,1> * m_triplets = new clever::vector<uint2,1>(triplets, nTriplets, ctx);
	std::cout << "done[" << m_triplets->get_count()  << "]" << std::endl;

	return m_triplets;
}
