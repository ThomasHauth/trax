#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class TripletThetaPhiPredictor: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	TripletThetaPhiPredictor(clever::context & ctext) :
		ctx(ctext),
		tripletThetaPhiPredict(ctext),
		tripletThetaPhiPredictStore(ctext)
{
		// create the buffers this algorithm will need to run
}

	clever::vector<uint2,1> * run(HitCollectionTransfer & hits, int nThreads, int layers[], int hitCount[],
			float dThetaWindow, float dPhiWindow, const clever::vector<uint2,1> & pairs)
	{

		int nLayer1 = hitCount[layers[0]-1];
		int nLayer2 = hitCount[layers[1]-1];
		int nLayer3 = hitCount[layers[2]-1];

		int nMaxTriplets = pairs.get_count() * nLayer3;

		std::cout << "Initializing oracle...";
		clever::vector<uint, 1> m_oracle(0, std::ceil(nMaxTriplets / 32.0), ctx);
		std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

		std::cout << "Initializing prefix sum...";
		clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
		std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		std::cout << "Running check kernel...";
		tripletThetaPhiPredict.run(
				//configuration
				dThetaWindow, dPhiWindow,
				pairs.get_count(),
				// input
				pairs.get_mem(), (nLayer1+nLayer2), nLayer3,
				hits.buffer(GlobalX()), hits.buffer(GlobalY()), hits.buffer(GlobalZ()),
				// output
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		std::cout << "Fetching prefix sum...";
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

		std::cout << "Fetching oracle...";
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
		//TODO implement prefix sum as kernel
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

		std::cout << "Storing prefix sum...";
		transfer::upload(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;

		int nCandidateTriplets = prefixSum[nThreads]; //we allocated nThreads+1 so total sum is in prefixSum[nThreads]
		std::cout << "Initializing triplet candidates...";
		clever::vector<uint2, 1> * m_triplets = new clever::vector<uint2, 1>(ctx, nCandidateTriplets);
		std::cout << "done[" << m_triplets->get_count()  << "]" << std::endl;

		std::cout << "Running store kernel...";
		tripletThetaPhiPredictStore.run(
				//configuration
				pairs.get_count(), (nLayer1+nLayer2), nLayer3,
				//input
				pairs.get_mem(),
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//output
				// output
				m_triplets->get_mem(),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		std::cout << "Fetching triplets...";
		std::vector<uint2> triplets(nCandidateTriplets);
		transfer::download(*m_triplets, triplets, ctx);
		std::cout <<"done[" << triplets.size() << "]" << std::endl;

#ifdef DEBUG_OUT
		std::cout << "Triplet Candidates:" << std::endl;
		for(uint2 i : triplets){
			std::cout << i.x << "-" << i.y << std::endl;
		}
#endif

		return m_triplets;
	}

	KERNEL11_CLASS( tripletThetaPhiPredict, double, double, int,  cl_mem, int, int, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
			__kernel void tripletThetaPhiPredict(
					//configuration
					double dThetaWindow, double dPhiWindow, int nPairs,
					// hit input
					__global const uint2 * pairs, int offset, int nThirdHits,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global uint * prefixSum )
	{
		size_t id = get_global_id( 0 );
		size_t threads = get_global_size( 0 );

		int workload = nPairs / threads + 1;
		int i = id * workload;
		int end = min(i + workload, nPairs); // for last thread, if not a full workload is present
		int nFound = 0;

		//printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nPairs);

		for(; i < end; ++i){ //workload loop

			int firstHit = pairs[i].x;
			int secondHit = pairs[i].y;

			//tanTheta1
			float theta = atan2(sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit])*(hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit])*(hitGlobalY[secondHit] - hitGlobalY[firstHit]))
																		, ( hitGlobalZ[secondHit] - hitGlobalZ[firstHit] ));
			float thetaLow = (1-dThetaWindow) * theta;
			float thetaHigh = (1+dThetaWindow) * theta;

			//TODO detector geometry
			float dr = 13;

			float zLow = hitGlobalZ[secondHit] + dr * tan(thetaLow);
			float zHigh = hitGlobalZ[secondHit] + dr * tan(thetaHigh);

			//tanPhi1
			float phi = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) , ( hitGlobalX[secondHit] - hitGlobalX[firstHit] ));

			float phiLow = (1-dPhiWindow) * phi;
			float phiHigh = (1+dPhiWindow) * phi;

			//loop over all third hits
			//TODO store hits in more suitable data structure, with phi pre-calculated and (z,phi) sorted
			for(int j = 0; j < nThirdHits; ++j){
				// check z range
				int index = offset+j;
				bool valid = zLow <= hitGlobalZ[index] && hitGlobalZ[index] <= zHigh;

				// check phi range
				float hPhi = atan2(hitGlobalY[index],hitGlobalX[index]);
				valid = valid * (phiLow <= hPhi && hPhi <= phiHigh);

				//if valid update nFound
				nFound = nFound + valid;

				//update oracle
				index = i*nThirdHits + j;
				atomic_or(&oracle[index / 32], (valid << (index % 32)));

				if(valid)
					printf("[ %lu ] Found valid candidate %i (%i-%i-%i). Word %i Bit %i\n", id, index, firstHit, secondHit, j, index / 32, index % 32);
			} // end hit loop

		} //end workload loop

		prefixSum[id] = nFound;
	});

	KERNEL7_CLASS( tripletThetaPhiPredictStore, int, int, int, cl_mem, cl_mem, cl_mem, cl_mem,
				__kernel void tripletThetaPhiPredictStore(
						//configuration
						int nPairs, int offset, int nThirdHits,
						// hit input
						__global const uint2 * pairs,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output triplet candidates
						__global uint2 * triplets)
		{
			size_t id = get_global_id( 0 );
			size_t threads = get_global_size( 0 );

			int workload = nPairs / threads + 1;
			int i = id * workload;
			int end = min(i + workload, nPairs); // for last thread, if not a full workload is present

			int pos = prefixSum[id];

			for(; i < end; ++i){

				for(int j = 0; j < nThirdHits; ++j){

					//is this a valid triplet?
					int index = i*nThirdHits+j;
					bool valid = oracle[index / 32] & (1 << (index % 32));

					//last triplet written on [pos] is valid one
					index = offset + j;
					triplets[pos].x = valid * i;
					triplets[pos].y = valid * index;

					//if(valid)
					//	printf("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

					//advance pos if valid
					pos = pos + valid;
				}
			}
		});

};
