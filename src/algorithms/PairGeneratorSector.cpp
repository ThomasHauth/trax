#include "PairGeneratorSector.h"
#include <algorithms/PrefixSum.h>


Pairing * PairGeneratorSector::run(HitCollection & hits,
				uint nThreads, const TripletConfigurations & layerTriplets, const Grid & grid)
		{

	std::vector<uint> oracleOffset;
	uint totalMaxPairs = 0;

	uint nLayerTriplets = layerTriplets.size();
	for(uint e = 0; e < grid.config.nEvents; ++e){
		for(uint p = 0; p < nLayerTriplets; ++p){

			TripletConfiguration layerPair(layerTriplets, p);

			LayerGrid layer1(grid, layerPair.layer1(),e);
			LayerGrid layer2(grid, layerPair.layer2(),e);

			uint nMaxPairs = layer1.size()*layer2.size();
			nMaxPairs = 32 * std::ceil(nMaxPairs / 32.0); //round to next multiple of 32

			oracleOffset.push_back(totalMaxPairs);
			totalMaxPairs += nMaxPairs;
		}
	}

	LOG << "Initializing oracle offsets for pair gen...";
	clever::vector<uint, 1> m_oracleOffset(oracleOffset, ctx);
	LOG << "done[" << m_oracleOffset.get_count()  << "]" << std::endl;

	LOG << "Initializing oracle for pair gen...";
	clever::vector<uint, 1> m_oracle(0, std::ceil(totalMaxPairs / 32.0), ctx);
	LOG << "done[" << m_oracle.get_count()  << "]" << std::endl;

	LOG << "Initializing prefix sum for pair gen...";
	clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*nLayerTriplets*nThreads+1, ctx);
	LOG << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

	//ctx.select_profile_event(KERNEL_COMPUTE_EVT());

	LOG << "Running pair gen kernel...";
	cl_event evt = pairCount.run(
			//configuration
			layerTriplets.transfer.buffer(Layer1()), layerTriplets.transfer.buffer(Layer2()), grid.config.nLayers,
			grid.transfer.buffer(Boundary()),
			grid.config.MIN_Z, grid.config.sectorSizeZ(),	grid.config.nSectorsZ,
			grid.config.MIN_PHI, grid.config.sectorSizePhi(), grid.config.nSectorsPhi,
			layerTriplets.transfer.buffer(pairSpreadZ()), layerTriplets.transfer.buffer(pairSpreadPhi()),
			// hit input
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			// intermeditate data: oracle for hit pairs, prefix sum for found pairs
			m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
			//local
			local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	PairGeneratorSector::events.push_back(evt);
	LOG << "done" << std::endl;

	if(PROLIX){
		PLOG << "Fetching prefix sum for pair gen...";
		std::vector<uint> vPrefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,vPrefixSum,ctx);
		PLOG << "done" << std::endl;

		PLOG << "Prefix sum: ";
		for(auto i : vPrefixSum){
			PLOG << i << " ; ";
		}
		PLOG << std::endl;
	}


	if(PROLIX){
		PLOG << "Fetching oracle for pair gen...";
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
	evt = prefixSum.run(m_prefixSum.get_mem(), m_prefixSum.get_count(), nThreads, PairGeneratorSector::events);
	uint nFoundPairs;
	transfer::downloadScalar(m_prefixSum, nFoundPairs, ctx, true, m_prefixSum.get_count()-1, 1, &evt);

	if(PROLIX){
		PLOG << "Fetching prefix sum for pair gen...";
		std::vector<uint> vPrefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,vPrefixSum,ctx);
		PLOG << "done" << std::endl;

		PLOG << "Prefix sum: ";
		for(auto i : vPrefixSum){
			PLOG << i << " ; ";
		}
		PLOG << std::endl;
	}

	LOG << "Initializing pairs...";
	Pairing * hitPairs = new Pairing(ctx, nFoundPairs, grid.config.nEvents, layerTriplets.size());
	LOG << "done[" << hitPairs->pairing.get_count()  << "]" << std::endl;


	LOG << "Running pair gen store kernel...";
	evt = pairStore.run(
			//configuration
			layerTriplets.transfer.buffer(Layer1()), layerTriplets.transfer.buffer(Layer2()), grid.config.nLayers,
			//grid
			grid.transfer.buffer(Boundary()), grid.config.nSectorsZ, grid.config.nSectorsPhi,
			// input for oracle and prefix sum
			m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
			// output of pairs
			hitPairs->pairing.get_mem(), hitPairs->pairingOffsets.get_mem(),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	PairGeneratorSector::events.push_back(evt);
	LOG << "done" << std::endl;

	if(PROLIX){
		PLOG << "Fetching pairs...";
		std::vector<uint2> pairs = hitPairs->getPairings();
		PLOG <<"done[" << pairs.size() << "]" << std::endl;

		PLOG << "Pairs:" << std::endl;
		for(uint i = 0; i < nFoundPairs; ++i){
			PLOG << "[" << i << "] "  << pairs[i].x << "-" << pairs[i].y << std::endl;
		}

		PLOG << "Fetching pair offets...";
		std::vector<uint> pairOffsets = hitPairs->getPairingOffsets();
		PLOG <<"done[" << pairOffsets.size() << "]" << std::endl;

		PLOG << "Pair Offsets:" << std::endl;
		for(uint i = 0; i < pairOffsets.size(); ++i){
			PLOG << "[" << i << "] "  << pairOffsets[i] << std::endl;
		}
	}

	return hitPairs;
}
