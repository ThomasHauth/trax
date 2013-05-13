/*
 * TripletThetaPhiPredictor.cpp
 *
 *  Created on: May 13, 2013
 *      Author: dfunke
 */

#include "TripletThetaPhiPredictor.h"

#define DEBUG_OUT
Pairing * TripletThetaPhiPredictor::run(HitCollection & hits, const DetectorGeometry & geom, const GeometrySupplement & geomSupplement, const Dictionary & dict,
			int nThreads, const LayerTriplets & layerTriplets, const Grid & grid,
			float dThetaWindow, float dPhiWindow, const Pairing & pairs){

		std::vector<uint> oracleOffset;
		uint totalMaxTriplets = 0;

		uint nLayerTriplets = layerTriplets.size();
		std::vector<uint> nFoundPairs = pairs.getPairingOffsets();

		for(uint e = 0; e < grid.config.nEvents; ++e){
			for(uint p = 0; p < nLayerTriplets; ++p){

				LayerTriplet layerTriplet(layerTriplets, p);


				LayerGrid layer3(grid, layerTriplet.layer3(),e);

				//plus 1 for offset shift [0] == 0 so nFoundHits for [0] is [1]
				uint nMaxTriplets = nFoundPairs[e * nLayerTriplets + p + 1]*layer3.size();
				nMaxTriplets = 32 * std::ceil(nMaxTriplets / 32.0); //round to next multiple of 32

				oracleOffset.push_back(totalMaxTriplets);
				totalMaxTriplets += nMaxTriplets;
			}
		}

		std::cout << "Initializing oracle offsets for triplet prediction...";
		clever::vector<uint, 1> m_oracleOffset(oracleOffset, ctx);
		std::cout << "done[" << m_oracleOffset.get_count()  << "]" << std::endl;

		std::cout << "Initializing oracle for prediction...";
		clever::vector<uint, 1> m_oracle(0, std::ceil(totalMaxTriplets / 32.0), ctx);
		std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

		std::cout << "Initializing prefix sum for prediction...";
		clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*nLayerTriplets*nThreads+1, ctx);
		std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		std::cout << "Running predict kernel...";
		cl_event evt = tripletThetaPhiPredict.run(
				//detector geometry
				geom.transfer.buffer(RadiusDict()), dict.transfer.buffer(Radius()), geomSupplement.transfer.buffer(MinRadius()), geomSupplement.transfer.buffer(MaxRadius()),
				grid.transfer.buffer(Boundary()), layerTriplets.transfer.buffer(Layer3()), grid.config.nLayers,
				grid.config.MIN_Z, grid.config.sectorSizeZ, grid.config.nSectorsZ,
				grid.config.MIN_PHI, grid.config.sectorSizePhi, grid.config.nSectorsPhi,
				//configuration
				dThetaWindow, dPhiWindow,
				// input
				pairs.pairing.get_mem(), pairs.pairingOffsets.get_mem(),
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
				hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()),
				// output
				m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
				local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
				//thread config
				range(nThreads, nLayerTriplets, grid.config.nEvents),
				range(nThreads, 1,1));
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum for prediction...";
		std::vector<uint> vPrefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,vPrefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : vPrefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

#ifdef DEBUG_OUT
		std::cout << "Fetching oracle for prediction...";
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
		uint nFoundTripletCandidates;
		transfer::downloadScalar(m_prefixSum, nFoundTripletCandidates, ctx, true, m_prefixSum.get_count()-1, 1, &evt);

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum for prediction...";
		transfer::download(m_prefixSum,vPrefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : vPrefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

		std::cout << "Initializing triplet candidates...";
		Pairing * m_triplets = new Pairing(ctx, nFoundTripletCandidates, grid.config.nEvents, layerTriplets.size());
		std::cout << "done[" << m_triplets->pairing.get_count()  << "]" << std::endl;

		std::cout << "Running predict store kernel...";
		evt = tripletThetaPhiPredictStore.run(
				//configuration
				grid.transfer.buffer(Boundary()), grid.config.nSectorsZ, grid.config.nSectorsPhi,
				layerTriplets.transfer.buffer(Layer3()), grid.config.nLayers,
				//input
				pairs.pairing.get_mem(), pairs.pairingOffsets.get_mem(),
				m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
				// output
				m_triplets->pairing.get_mem(), m_triplets->pairingOffsets.get_mem(),
				//thread config
				range(nThreads, nLayerTriplets, grid.config.nEvents),
				range(nThreads, 1,1));
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_STORE_EVT());


#ifdef DEBUG_OUT
		std::cout << "Fetching triplet candidates...";
		std::vector<uint2> cands = m_triplets->getPairings();
		std::cout <<"done[" << cands.size() << "]" << std::endl;

		std::cout << "Candidates:" << std::endl;
		for(uint i = 0; i < nFoundTripletCandidates; ++i){
			std::cout << "[" << i << "] "  << cands[i].x << "-" << cands[i].y << std::endl;
		}

		std::cout << "Fetching candidates offets...";
		std::vector<uint> candOffsets = m_triplets->getPairingOffsets();
		std::cout <<"done[" << candOffsets.size() << "]" << std::endl;

		std::cout << "Candidate Offsets:" << std::endl;
		for(uint i = 0; i < candOffsets.size(); ++i){
			std::cout << "[" << i << "] "  << candOffsets[i] << std::endl;
		}
#endif

		return m_triplets;
	}

