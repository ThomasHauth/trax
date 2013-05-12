#include "PairGeneratorSector.h"
#include <algorithms/PrefixSum.h>

#define DEBUG_OUT
clever::vector<uint2,1> * PairGeneratorSector::run(HitCollection & hits,
				uint nThreads, const LayerTriplets & layerTriplets, const Grid & grid,
				uint spreadZ, uint spreadPhi)
		{

	std::vector<uint> oracleOffset;
	uint totalMaxPairs = 0;

	uint nLayerTriplets = layerTriplets.size();
	for(uint e = 0; e < grid.config.nEvents; ++e){
		for(uint p = 0; p < nLayerTriplets; ++p){

			LayerTriplet layerPair(layerTriplets, p);

			LayerGrid layer1(grid, layerPair.layer1(),e);
			LayerGrid layer2(grid, layerPair.layer2(),e);

			uint nMaxPairs = layer1.size()*layer2.size();
			nMaxPairs = 32 * std::ceil(nMaxPairs / 32.0); //round to next multiple of 32

			oracleOffset.push_back(totalMaxPairs);
			totalMaxPairs += nMaxPairs;
		}
	}

	std::cout << "Initializing oracle offsets for pair gen...";
	clever::vector<uint, 1> m_oracleOffset(oracleOffset, ctx);
	std::cout << "done[" << m_oracleOffset.get_count()  << "]" << std::endl;

	std::cout << "Initializing oracle for pair gen...";
	clever::vector<uint, 1> m_oracle(0, std::ceil(totalMaxPairs / 32.0), ctx);
	std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

	std::cout << "Initializing prefix sum for pair gen...";
	clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*nLayerTriplets*nThreads+1, ctx);
	std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

	//ctx.select_profile_event(KERNEL_COMPUTE_EVT());

	std::cout << "Running pair gen kernel...";
	cl_event evt = pairSectorGen.run(
			//configuration
			layerTriplets.transfer.buffer(Layer1()), layerTriplets.transfer.buffer(Layer2()), grid.config.nLayers,
			grid.transfer.buffer(Boundary()),
			grid.config.MIN_Z, grid.config.sectorSizeZ,	grid.config.nSectorsZ,
			grid.config.MIN_PHI, grid.config.sectorSizePhi, grid.config.nSectorsPhi,
			spreadZ, spreadPhi,
			// hit input
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			// intermeditate data: oracle for hit pairs, prefix sum for found pairs
			m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
			//local
			local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	std::cout << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

#ifdef DEBUG_OUT
	std::cout << "Fetching prefix sum for pair gen...";
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
	std::cout << "Fetching oracle for pair gen...";
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
	uint nFoundPairs;
	transfer::downloadScalar(m_prefixSum, nFoundPairs, ctx, true, m_prefixSum.get_count()-1, 1, &evt);

#ifdef DEBUG_OUT
	std::cout << "Fetching prefix sum for pair gen...";
	transfer::download(m_prefixSum,vPrefixSum,ctx);
	std::cout << "done" << std::endl;

	std::cout << "Prefix sum: ";
	for(auto i : vPrefixSum){
		std::cout << i << " ; ";
	}
	std::cout << std::endl;
#endif
	std::cout << "Initializing pairs...";
	clever::vector<uint2, 1> * m_pairs = new clever::vector<uint2, 1>(ctx, nFoundPairs);
	std::cout << "done[" << m_pairs->get_count()  << "]" << std::endl;


	std::cout << "Running pair gen store kernel...";
	evt = pairStore.run(
			//configuration
			layerTriplets.transfer.buffer(Layer1()), layerTriplets.transfer.buffer(Layer2()), grid.config.nLayers,
			//grid
			grid.transfer.buffer(Boundary()), grid.config.nSectorsZ, grid.config.nSectorsPhi,
			// input for oracle and prefix sum
			m_oracle.get_mem(), m_oracleOffset.get_mem(), m_prefixSum.get_mem(),
			// output of pairs
			m_pairs->get_mem(),
			//thread config
			range(nThreads, nLayerTriplets, grid.config.nEvents),
			range(nThreads, 1,1));
	std::cout << "done" << std::endl;

	ctx.add_profile_event(evt, KERNEL_STORE_EVT());

#ifdef DEBUG_OUT
	std::cout << "Fetching pairs...";
	std::vector<uint2> pairs(nFoundPairs);
	transfer::download(*m_pairs, pairs, ctx);
	std::cout <<"done[" << pairs.size() << "]" << std::endl;

	std::cout << "Pairs:" << std::endl;
	for(uint i = 0; i < nFoundPairs; ++i){
		std::cout << "[" << i << "] "  << pairs[i].x << "-" << pairs[i].y << std::endl;
	}
#endif

	return m_pairs;
		}
