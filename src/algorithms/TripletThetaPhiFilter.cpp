/*
 * TripletThetaPhiFilter.cpp
 *
 *  Created on: May 13, 2013
 *      Author: dfunke
 */

#include "TripletThetaPhiFilter.h"

TrackletCollection * TripletThetaPhiFilter::run(HitCollection & hits, const Grid & grid,
		const Pairing & pairs, const Pairing & tripletCandidates,
		int nThreads, const TripletConfigurations & layerTriplets)
{
	std::vector<uint> oracleOffset;
	uint totalCandidates = 0;

	uint nLayerTriplets = layerTriplets.size();
	std::vector<uint> nFoundCandidates = tripletCandidates.getPairingOffsets();

	for(uint e = 0; e < grid.config.nEvents; ++e){
		for(uint p = 0; p < nLayerTriplets; ++p){

			//plus 1 for offset shift [0] == 0 so nFoundHits for [0] is [1] - [0]
			uint candidates = nFoundCandidates[e * nLayerTriplets + p + 1] - nFoundCandidates[e * nLayerTriplets + p];
			candidates = 32 * std::ceil(candidates / 32.0); //round to next multiple of 32

			oracleOffset.push_back(totalCandidates);
			totalCandidates += candidates;
		}
	}

	LOG << "Initializing oracle offsets for triplet prediction...";
	clever::vector<uint, 1> m_oracleOffset(oracleOffset, ctx);
	LOG << "done[" << m_oracleOffset.get_count()  << "]" << std::endl;

	LOG << "Initializing oracle...";
	clever::vector<uint, 1> m_oracle(0, std::ceil(totalCandidates / 32.0), ctx);
	LOG << "done[" << m_oracle.get_count()  << "]" << std::endl;

	LOG << "Initializing prefix sum...";
	clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*nLayerTriplets*nThreads+1, ctx);
	LOG << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

	LOG << "Running filter kernel...";
	cl_event evt = tripletThetaPhiCheck.run(
			//configuration
			layerTriplets.transfer.buffer(dThetaCut()), layerTriplets.transfer.buffer(dPhiCut()), layerTriplets.transfer.buffer(tipCut()),
			// input
			pairs.pairing.get_mem(),
			tripletCandidates.pairing.get_mem(), tripletCandidates.pairingOffsets.get_mem(),
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			// output
			m_oracle.get_mem(),m_oracleOffset.get_mem(),  m_prefixSum.get_mem(),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	LOG << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

	if(PROLIX){
		PLOG << "Fetching prefix sum...";
		std::vector<uint> cPrefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,cPrefixSum,ctx);
		PLOG << "done" << std::endl;
		PLOG << "Prefix sum: ";
		for(auto i : cPrefixSum){
			PLOG << i << " ; ";
		}
		PLOG << std::endl;
	}

	if(PROLIX){
		PLOG << "Fetching oracle...";
		std::vector<uint> oracle(m_oracle.get_count());
		transfer::download(m_oracle,oracle,ctx);
		PLOG << "done" << std::endl;
		PLOG << "Oracle: ";
		for(auto i : oracle){
			PLOG << i << " ; ";
		}
		PLOG << std::endl;
	}

	//Calculate prefix sum
	PrefixSum prefixSum(ctx);
	evt = prefixSum.run(m_prefixSum.get_mem(), m_prefixSum.get_count(), nThreads);
	uint nFoundTriplets;
	transfer::downloadScalar(m_prefixSum, nFoundTriplets, ctx, true, m_prefixSum.get_count()-1, 1, &evt);


	if(PROLIX){
		PLOG << "Fetching prefix sum...";
		std::vector<uint> cPrefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,cPrefixSum,ctx);
		PLOG << "done" << std::endl;
		PLOG << "Prefix sum: ";
		for(auto i : cPrefixSum){
			PLOG << i << " ; ";
		}
		PLOG << std::endl;
	}

	TrackletCollection * tracklets = new TrackletCollection(nFoundTriplets, grid.config.nEvents, layerTriplets.size(), ctx);
	LOG << "Reserving space for " << nFoundTriplets << " tracklets" << std::endl;

	tracklets->transfer.initBuffers(ctx, *tracklets);

	LOG << "Running filter store kernel...";
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
	LOG << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_STORE_EVT());

	LOG << "Fetching triplets...";
	tracklets->transfer.fromDevice(ctx, *tracklets);
	LOG <<"done[" << tracklets->size() << "]" << std::endl;

	if(PROLIX){
		PLOG << "Fetching triplet offets...";
		std::vector<uint> tripOffsets = tracklets->getTrackletOffsets();
		PLOG <<"done[" << tripOffsets.size() << "]" << std::endl;

		PLOG << "Tracklet Offsets:" << std::endl;
		for(uint i = 0; i < tripOffsets.size(); ++i){
			PLOG << "[" << i << "] "  << tripOffsets[i] << std::endl;
		}
	}

	return tracklets;
}
