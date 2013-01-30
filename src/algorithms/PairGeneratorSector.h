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

	clever::vector<uint2,1> * run(HitCollectionTransfer & hits,
				int nThreads, int layers[], const LayerSupplement & layerSupplement,
				int nSectors)
		{

			/*int nLayer1 = layerSupplement[layers[0]-1].nHits;
			int nLayer2 = layerSupplement[layers[1]-1].nHits;

			int nMaxPairs = nLayer1 * nLayer2;

			std::cout << "Transferring sector borders...";

			//border layer 1
			clever::vector<uint, 1> m_borders1(layerSupplement[layers[0]-1].sectorBorders,ctx);
			clever::vector<uint, 1> m_borders2(layerSupplement[layers[1]-1].sectorBorders,ctx);

			std::cout << "done[" << m_borders1.get_count() + m_borders2.get_count() << "]" << std::endl;

			std::cout << "Initializing oracle for pair gen...";
			clever::vector<uint, 1> m_oracle(0, std::ceil(nMaxPairs / 32.0), ctx);
			std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

			std::cout << "Initializing prefix sum for pair gen...";
			clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
			std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

			//ctx.select_profile_event(KERNEL_COMPUTE_EVT());

			std::cout << "Running pair gen kernels...";
			//for(int sector = 1; sector <= nSectors; ++sector){
				pairSectorGen.run(
						//configuration
						3, nSectors,
						nLayer1, nLayer2,
						// hit input
						m_borders1.get_mem(), m_borders2.get_mem(),
						// intermeditate data: oracle for hit pairs, prefix sum for found pairs
						m_oracle.get_mem(), m_prefixSum.get_mem(),
						//thread config
						nThreads);
			//}
			std::cout << "done" << std::endl;

			std::cout << "Fetching prefix sum for pair gen...";
			std::vector<uint> prefixSum(m_prefixSum.get_count());
			transfer::download(m_prefixSum,prefixSum,ctx);
			std::cout << "done" << std::endl;

	#ifdef DEBUG_OUT
			std::cout << "Prefix sum: ";
			for(auto i : prefixSum){
				std::cout << i << " ; ";
			}
			std::cout << std::endl;
	#endif

			std::cout << "Fetching oracle for pair gen...";
			std::vector<uint> oracle(m_oracle.get_count());
			transfer::download(m_oracle,oracle,ctx);
			std::cout << "done" << std::endl;

	#ifdef DEBUG_OUT
			std::cout << "Oracle: ";
			for(auto i : oracle){
				std::cout << i << " ; ";
			}
			std::cout << std::endl;
	#endif

			//Calculate prefix sum
			//TODO[gpu] implement prefix sum as kernel
			uint s = 0;
			for(uint i = 0; i < prefixSum.size(); ++i){
				int tmp = s;
				s += prefixSum[i];
				prefixSum[i] = tmp;
			}

	#ifdef DEBUG_OUT
			std::cout << "Prefix sum: ";
			for(auto i : prefixSum){
				std::cout << i << " ; ";
			}
			std::cout << std::endl;
	#endif

			std::cout << "Storing prefix sum for pair gen...";
			transfer::upload(m_prefixSum,prefixSum,ctx);
			std::cout << "done" << std::endl;

			int nPairs = prefixSum[nThreads]; //we allocated nThreads+1 so total sum is in prefixSum[nThreads]
			std::cout << "Initializing pairs...";
			clever::vector<uint2, 1> * m_pairs = new clever::vector<uint2, 1>(ctx, nPairs);
			std::cout << "done[" << m_pairs->get_count()  << "]" << std::endl;

			//ctx.select_profile_event(KERNEL_STORE_EVT());

			std::cout << "Running pair gen store kernel...";
			pairStore.run(
					//configuration
					nLayer1, nLayer2,
					layerSupplement[layers[0]-1].offset, layerSupplement[layers[1]-1].offset,
					// input for oracle and prefix sum
					m_oracle.get_mem(), m_prefixSum.get_mem(),
					// output of pairs
					m_pairs->get_mem(),
					//thread config
					nThreads);
			std::cout << "done" << std::endl;

			std::cout << "Fetching pairs...";
			std::vector<uint2> pairs(nPairs);
			transfer::download(*m_pairs, pairs, ctx);
			std::cout <<"done[" << pairs.size() << "]" << std::endl;

	#ifdef DEBUG_OUT
			std::cout << "Pairs:" << std::endl;
			for(uint2 i : pairs){
				std::cout << i.x << "-" << i.y << std::endl;
			}
	#endif

			return m_pairs;*/
		}

	KERNEL8_CLASS( pairSectorGen, uint, uint, uint, uint, cl_mem, cl_mem,  cl_mem, cl_mem,
			__kernel void pairSectorGen(
					//configuration
					uint sector, uint nSectors,
					uint nLayer1, uint nLayer2,
					// hit input
					__global const uint * sectorBorders1, __global const uint * sectorBorders2,
					// intermeditate data: oracle for hit pairs, prefix sum for found pairs
					__global uint * oracle, __global uint * prefixSum )
	{
		const size_t gid = get_global_id( 0 );
		const size_t lid = get_local_id( 0 );
		const size_t threads = get_global_size( 0 );

		uint workload = (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)]) / threads + 1;
		uint i = gid * workload;
		uint end = min(i + workload, (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)])); // for last thread, if not a full workload is present
		uint nFound = 0;

		printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", gid, threads, workload, i, end, (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)]));

		//last used as tempory variable
		int last = (2*(sector-1)-1) % nSectors; //signed for -1 underflow
		int j = last + (last < 0) * nSectors;
		j = sign(last) * sectorBorders2[j]; //wrap around through neg. number

		//last used with real intention
		last = sectorBorders2[(2*sector+1) % nSectors];
		j *= sign(last-j); //if warp around then end2 < j => -1

		for(; i < end; ++i){

			for(; j < last; j += sign(j) * 1){

				if(abs(j) == nLayer2) //comparison between signed and unsigned
					j = 0;

				++nFound;

				//update oracle
				uint index = i*nLayer2 + abs(j);
				atomic_or(&oracle[index / 32], (1 << (index % 32)));

			} // end second hit loop

		} // end workload loop

		prefixSum[gid] += nFound;
	});

	KERNEL7_CLASS( pairStore, uint, uint, uint, uint, cl_mem, cl_mem, cl_mem,
				__kernel void pairStore(
						//configuration
						uint nLayer1, uint nLayer2,
						uint offset1, uint offset2,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output of pairs
						__global uint2 * pairs)
		{
			size_t id = get_global_id( 0 );
			size_t threads = get_global_size( 0 );

			uint workload = (nLayer1) / threads + 1;
			uint i = id * workload;
			uint end = min(i + workload, nLayer1); // for last thread, if not a full workload is present

			uint pos = prefixSum[id];

			for(; i < end; ++i){

				for(uint j = 0; j < nLayer2 && pos < prefixSum[id+1]; ++j){ // pos < prefixSum[id+1] can lead to thread divergence

					//is this a valid triplet?
					uint index = i*nLayer2+j;
					bool valid = oracle[index / 32] & (1 << (index % 32));

					//last triplet written on [pos] is valid one
					pairs[pos].x = valid * (offset1 + i);
					pairs[pos].y = valid * (offset2 + j);

					//if(valid)
					//	printf("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

					//advance pos if valid
					pos = pos + valid;
				}
			}
		});

};
