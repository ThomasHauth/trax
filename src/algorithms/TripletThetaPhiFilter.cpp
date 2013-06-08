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

	uint nTripletCandidates = tripletCandidates.pairing.get_count();
	uint nOracleCount = std::ceil(nTripletCandidates / 32.0);
	uint nGroups = (uint) std::max(1.0f, ceil(((float) nTripletCandidates)/nThreads));

	LOG << "Initializing oracle...";
	clever::vector<uint, 1> m_oracle(0, nOracleCount, ctx);
	LOG << "done[" << m_oracle.get_count()  << "]" << std::endl;

	LOG << "Running filter kernel...";
	cl_event evt = filterCount.run(
			//configuration
			layerTriplets.transfer.buffer(dThetaCut()), layerTriplets.transfer.buffer(dPhiCut()), layerTriplets.transfer.buffer(tipCut()),
			// input
			pairs.pairing.get_mem(),
			tripletCandidates.pairing.get_mem(), nTripletCandidates,
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			hits.transfer.buffer(EventNumber()), hits.transfer.buffer(DetectorLayer()),
			// output
			m_oracle.get_mem(),
			//thread config
			range(nGroups * nThreads),
			range(nThreads));
	TripletThetaPhiFilter::events.push_back(evt);
	LOG << "done" << std::endl;

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

	LOG << "Initializing prefix sum...";
	clever::vector<uint, 1> m_prefixSum(0, nOracleCount+1, ctx);
	LOG << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

	LOG << "Running popcount kernel...";
	nGroups = (uint) std::max(1.0f, ceil(((float) nOracleCount)/nThreads));
	evt = filterPopCount.run(
			m_oracle.get_mem(), m_prefixSum.get_mem(), nOracleCount,
			//threads
			range(nGroups * nThreads),
			range(nThreads));
	TripletThetaPhiFilter::events.push_back(evt);
	LOG << "done" << std::endl;

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

	//Calculate prefix sum
	PrefixSum prefixSum(ctx);
	evt = prefixSum.run(m_prefixSum.get_mem(), m_prefixSum.get_count(), nThreads, TripletThetaPhiFilter::events);
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
	evt = filterStore.run(
			//input
			pairs.pairing.get_mem(), layerTriplets.size(),
			tripletCandidates.pairing.get_mem(), nTripletCandidates,
			m_oracle.get_mem(), nOracleCount, m_prefixSum.get_mem(),
			// output
			tracklets->transfer.buffer(TrackletHit1()), tracklets->transfer.buffer(TrackletHit2()), tracklets->transfer.buffer(TrackletHit3()),
			tracklets->trackletOffsets.get_mem(),
			//thread config
			range(nGroups * nThreads), //same number of groups as for popcount kernel
			range(nThreads));
	TripletThetaPhiFilter::events.push_back(evt);
	LOG << "done" << std::endl;

	LOG << "Running filter offset store kernel...";
	nGroups = (uint) std::max(1.0f, ceil(((float) nFoundTriplets)/nThreads));
	evt = filterOffsetStore.run(
			//tracklets
			tracklets->transfer.buffer(TrackletHit1()), tracklets->transfer.buffer(TrackletHit2()), tracklets->transfer.buffer(TrackletHit3()),
			nFoundTriplets, layerTriplets.size(),
			hits.transfer.buffer(EventNumber()), hits.transfer.buffer(DetectorLayer()),
			//output
			tracklets->trackletOffsets.get_mem(),
			range(nGroups * nThreads),
			range(nThreads));
	TripletThetaPhiFilter::events.push_back(evt);
	LOG << "done" << std::endl;

	LOG << "Running filter offset monotonize kernel...";
	nGroups = (uint) std::max(1.0f, ceil(((float) tracklets->trackletOffsets.get_count())/nThreads));
	evt = filterOffsetMonotonizeStore.run(
			tracklets->trackletOffsets.get_mem(), tracklets->trackletOffsets.get_count(),
			range(nGroups * nThreads),
			range(nThreads));
	TripletThetaPhiFilter::events.push_back(evt);
	LOG << "done" << std::endl;

	LOG << "Fetching triplets...";
	tracklets->transfer.fromDevice(ctx, *tracklets);
	LOG <<"done[" << tracklets->size() << "]" << std::endl;

	if(PROLIX){
		PLOG << "Tracklets: " << std::endl;
		for(uint i = 0; i < nFoundTriplets; ++i){
			Tracklet tracklet(*tracklets, i);
			PLOG << "[" << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3() << "]" << std::endl;;
		}
	}

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
