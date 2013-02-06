#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class PairGeneratorSector: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	PairGeneratorSector(clever::context & ctext) :
		ctx(ctext),
		pairSectorGen(ctext),
		pairStore(ctext)
{
		// create the buffers this algorithm will need to run
#ifdef DEBUG_OUT
		std::cout << "FilterKernel WorkGroupSize: " << pairSectorGen.getWorkGroupSize() << std::endl;
		std::cout << "StoreKernel WorkGroupSize: " << pairSectorStore.getWorkGroupSize() << std::endl;
#endif
}

	static std::string KERNEL_COMPUTE_EVT() {return "PairGeneratorSector_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "PairGeneratorSector_STORE";}

	clever::vector<uint2,1> * run(HitCollection & hits,
				int nThreads, int layers[], const LayerSupplement & layerSupplement, const Grid & grid,
				int spreadZ)
		{

			int nLayer1 = layerSupplement[layers[0]-1].getNHits();
			int nLayer2 = layerSupplement[layers[1]-1].getNHits();
			int nMaxPairs = nLayer1 * nLayer2;

			std::cout << "Initializing oracle for pair gen...";
			clever::vector<uint, 1> m_oracle(0, std::ceil(nMaxPairs / 32.0), ctx);
			std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

			std::cout << "Initializing prefix sum for pair gen...";
			clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
			std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

			//ctx.select_profile_event(KERNEL_COMPUTE_EVT());

			std::cout << "Running pair gen kernel...";
			cl_event evt = pairSectorGen.run(
						//configuration
						layers[0], layers[1],
						layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()),
						grid.transfer.buffer(Boundary()), grid.config.MIN_Z, grid.config.sectorSizeZ,
						grid.config.nSectorsZ, grid.config.nSectorsPhi,
						spreadZ,
						// hit input
						hits.transfer.buffer(GlobalZ()), hits.transfer.buffer(HitId()),
						// intermeditate data: oracle for hit pairs, prefix sum for found pairs
						m_oracle.get_mem(), m_prefixSum.get_mem(),
						//thread config
						nThreads);
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
			PrefixSum PrefixSum(ctx);
			int nFoundPairs = PrefixSum.run(m_prefixSum, nThreads, true);

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
					layers[0], layers[1],
					layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()),
					// input for oracle and prefix sum
					m_oracle.get_mem(), m_prefixSum.get_mem(),
					// output of pairs
					m_pairs->get_mem(),
					//thread config
					nThreads);
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

	KERNEL14_CLASS( pairSectorGen, uint, uint, cl_mem, cl_mem,  cl_mem, cl_float, cl_float, uint, uint, uint, cl_mem, cl_mem, cl_mem, cl_mem,
			__kernel void pairSectorGen(
					//configuration
					uint layer1, uint layer2, __global const uint * layerHits, __global const uint * layerOffsets,
					__global const uint * sectorBoundaries, float minZ, float sectorSizeZ , uint nSectorsZ, uint nSectorsPhi,
					//how many sectors to cover
					uint spread,
					//hit z
					__global const float * hitGlobalZ, __global const uint * hitId,
					__global uint * oracle, __global uint * prefixSum )
	{
		const size_t gid = get_global_id( 0 );
		const size_t threads = get_global_size( 0 );

		uint hits1 = layerHits[layer1-1];
		uint offset1 = layerOffsets[layer1-1];

		uint hits2 = layerHits[layer2-1];
		uint offset2 = layerOffsets[layer2-1];

		uint workload = hits1 / threads + 1;
		uint i = gid * workload;
		uint end = min(i + workload, hits1); // for last thread, if not a full workload is present
		uint nFound = 0;

		//printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", gid, threads, workload, i, end, (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)]));

		for(; i < end; ++i){

			uint minSectorBorder = floor((hitGlobalZ[offset1 + i] - minZ) / sectorSizeZ); //register used to store sector
			uint maxSectorBorder = min(minSectorBorder + spread+1, nSectorsZ); // upper sector border equals sectorNumber + 1
			minSectorBorder = max(0u, minSectorBorder-spread); //lower sector border equals sectorNumber

			for(uint j = sectorBoundaries[(layer2-1)*(nSectorsZ+1)*(nSectorsPhi+1) + minSectorBorder*(nSectorsPhi+1)];
					j < sectorBoundaries[(layer2-1)*(nSectorsZ+1)*(nSectorsPhi+1) + maxSectorBorder*(nSectorsPhi+1)];
					++j){

				++nFound;

				//update oracle
				uint index = i*hits2 + j;
				atomic_or(&oracle[index / 32], (1 << (index % 32)));

			} // end second hit loop

		} // end workload loop

		prefixSum[gid] = nFound;

		//printf("[%lu] Found %u pairs\n", gid, nFound);
	});

	KERNEL7_CLASS( pairStore, uint, uint, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
				__kernel void pairStore(
						//configuration
						uint layer1, uint layer2, __global const uint * layerHits, __global const uint * layerOffsets,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output of pairs
						__global uint2 * pairs)
		{
		const size_t gid = get_global_id( 0 );
		const size_t threads = get_global_size( 0 );

		uint hits1 = layerHits[layer1-1];
		uint offset1 = layerOffsets[layer1-1];

		uint hits2 = layerHits[layer2-1];
		uint offset2 = layerOffsets[layer2-1];

		uint workload = hits1 / threads + 1;
		uint i = gid * workload;
		uint end = min(i + workload, hits1); // for last thread, if not a full workload is present

		uint pos = prefixSum[gid];

		for(; i < end; ++i){
			for(uint j = 0; j < hits2 && pos < prefixSum[gid+1]; ++j){ // pos < prefixSum[id+1] can lead to thread divergence

				//is this a valid triplet?
				uint index = i * hits2 + j;
				bool valid = oracle[index / 32] & (1 << (index % 32));

				//last triplet written on [pos] is valid one
				pairs[pos].x = valid * (offset1 + i);
				pairs[pos].y = valid * (offset2 + j);

				//if(valid)
				//printf("[ %lu ] Written at %i: %i-%i\n", gid, pos, pairs[pos].x,pairs[pos].y);

				//advance pos if valid
				pos += valid;
			}
		}
		});

};
