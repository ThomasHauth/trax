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
class TripletThetaPhiFilter: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	TripletThetaPhiFilter(clever::context & ctext) :
		ctx(ctext),
		tripletThetaPhiCheck(ctext),
		tripletThetaPhiStore(ctext)
{
		// create the buffers this algorithm will need to run
}

	TrackletCollection * run(HitCollectionTransfer & hits, int nThreads, int layers[], int hitCount[], float dThetaCut, float dPhiCut)
	{

		int nLayer1 = hitCount[layers[0]-1];
		int nLayer2 = hitCount[layers[1]-1];
		int nLayer3 = hitCount[layers[2]-1];

		int nMaxPairs = nLayer1 * nLayer2;

		//TODO here we should use some clever pair prediction kernel
		std::vector<uint2> pairs;
		for(int i = 0; i < nLayer1; ++i)
			for(int j=0; j < nLayer2; ++j)
				pairs.push_back(uint2(i,nLayer1 + j));

		std::cout << "Transferring " << pairs.size() << " pairs...";
		clever::vector<uint2,1> m_pairs(pairs, nMaxPairs, ctx);
		int nPairs = m_pairs.get_count();
		std::cout << "done[" << nPairs  << "]" << std::endl;

		int nMaxTriplets = nPairs * nLayer3;

		//TODO here we should use some clever triplet candidate prediction kernel
		std::vector<uint2> triplets;
		for(int i = 0; i < nPairs; ++i)
			for(int j = 0; j < nLayer3; ++j)
				triplets.push_back(uint2(i,nLayer1 + nLayer2 + j));

		std::cout << "Transferring " << triplets.size() << " triplets...";
		clever::vector<uint2,1> m_triplets(triplets, nMaxTriplets, ctx);
		int nTriplets = m_triplets.get_count();
		std::cout << "done[" << nTriplets  << "]" << std::endl;

		std::cout << "Initializing oracle...";
		clever::vector<uint, 1> m_oracle(0, std::ceil(nTriplets / 32.0), ctx);
		std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

		std::cout << "Initializing prefix sum...";
		clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
		std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		std::cout << "Running check kernel...";
		tripletThetaPhiCheck.run(
				//configuration
				dThetaCut, dPhiCut,
				nTriplets,
				// input
				m_pairs.get_mem(), m_triplets.get_mem(),
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
		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;

		std::cout << "Fetching oracle...";
		std::vector<uint> oracle(m_oracle.get_count());
		transfer::download(m_oracle,oracle,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Oracle: ";
		for(auto i : oracle){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;

		//Calculate prefix sum
		//TODO implement prefix sum as kernel
		uint s = 0;
		for(uint i = 0; i < prefixSum.size(); ++i){
			int tmp = s;
			s += prefixSum[i];
			prefixSum[i] = tmp;
		}

		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
		std::cout << "Storing prefix sum...";
		transfer::upload(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;

		int nFoundTriplets = prefixSum[nThreads]; //we allocated nThreads+1 so total sum is in prefixSum[nThreads]
		TrackletCollection * tracklets = new TrackletCollection(nFoundTriplets);
		std::cout << "Reserving space for " << nFoundTriplets << " tracklets" << std::endl;

		TrackletCollectionTransfer clTrans_tracklet;
		clTrans_tracklet.initBuffers(ctx, *tracklets);

		std::cout << "Running store kernel...";
		tripletThetaPhiStore.run(
				//configuration
				nTriplets,
				//input
				m_pairs.get_mem(), m_triplets.get_mem(),
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//output
				// output
				clTrans_tracklet.buffer(TrackletHit1()), clTrans_tracklet.buffer(TrackletHit2()), clTrans_tracklet.buffer(TrackletHit3()),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		std::cout << "Fetching triplets...";
		clTrans_tracklet.fromDevice(ctx, *tracklets);
		std::cout <<"done[" << tracklets->size() << "]" << std::endl;

		return tracklets;
	}

	KERNEL10_CLASS( tripletThetaPhiCheck, double, double, int, cl_mem, cl_mem,  cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
			__kernel void tripletThetaPhiCheck(
					//configuration
					double dThetaCut, double dPhiCut, int nTriplets,
					// hit input
					__global const uint2 * pairs, __global const uint2 * triplets,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global uint * prefixSum )
	{
		size_t id = get_global_id( 0 );
		size_t threads = get_global_size( 0 );

		int workload = nTriplets / threads + 1;
		int i = id * workload;
		int end = min(i + workload, nTriplets); // for last thread, if not a full workload is present
		int nValid = 0;

		//printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nTriplets);

		for(; i < end; ++i){

			int firstHit = pairs[triplets[i].x].x;
			int secondHit = pairs[triplets[i].x].y;
			int thirdHit = triplets[i].y;
			bool valid = true;

			//tanTheta1
			float angle1 = atan2(sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit])*(hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit])*(hitGlobalY[secondHit] - hitGlobalY[firstHit]))
																		, ( hitGlobalZ[secondHit] - hitGlobalZ[firstHit] ));
			//tanTheta2
			float angle2 = atan2(sqrt((hitGlobalX[thirdHit] - hitGlobalX[secondHit])*(hitGlobalX[thirdHit] - hitGlobalX[secondHit])
					+ (hitGlobalY[thirdHit] - hitGlobalY[secondHit])*(hitGlobalY[thirdHit] - hitGlobalY[secondHit]))
																								, ( hitGlobalZ[thirdHit] - hitGlobalZ[secondHit] ));
			float ratio = angle2/angle1;
			valid = valid * (1-dThetaCut <= ratio && ratio <= 1+dThetaCut);

			//tanPhi1
			angle1 = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) , ( hitGlobalX[secondHit] - hitGlobalX[firstHit] ));
			//tanPhi2
			angle2 = atan2((hitGlobalY[thirdHit] - hitGlobalY[secondHit]) , ( hitGlobalX[thirdHit] - hitGlobalX[secondHit] ));

			ratio = angle2/angle1;
			valid = valid * (1-dPhiCut <= ratio && ratio <= 1+dPhiCut);

			//found good triplet?
			nValid = nValid + valid;

			//if(valid)
			//	printf("[ %lu ] Found valid track %i (%i-%i-%i). Word %i Bit %i\n", id, i, firstHit, secondHit, thirdHit, i / 32, i % 32);

			atomic_or(&oracle[i / 32], (valid << (i % 32)));

		}

		prefixSum[id] = nValid;
	});

	KERNEL8_CLASS( tripletThetaPhiStore, int, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
				__kernel void tripletThetaPhiStore(
						//configuration
						int nTriplets,
						// hit input
						__global const uint2 * pairs, __global const uint2 * triplets,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output of tracklet data
						__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3)
		{
			size_t id = get_global_id( 0 );
			size_t threads = get_global_size( 0 );

			int workload = nTriplets / threads + 1;
			int i = id * workload;
			int end = min(i + workload, nTriplets); // for last thread, if not a full workload is present

			int pos = prefixSum[id];

			for(; i < end; ++i){

				//is this a valid triplet?
				bool valid = oracle[i / 32] & (1 << (i % 32));

				//last triplet written on [pos] is valid one
				trackletHitId1[pos] = valid * pairs[triplets[i].x].x;
				trackletHitId2[pos] = valid * pairs[triplets[i].x].y;
				trackletHitId3[pos] = valid * triplets[i].y;

				//if(valid)
				//	printf("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

				//advance pos if valid
				pos = pos + valid;
			}
		});

};
