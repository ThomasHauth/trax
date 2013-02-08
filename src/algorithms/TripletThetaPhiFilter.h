#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/GeometrySupplement.h>

#include <algorithms/TripletThetaPhiPredictor.h>
#include <algorithms/PairGeneratorSector.h>
#include <algorithms/PrefixSum.h>

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
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "FilterKernel WorkGroupSize: " << tripletThetaPhiCheck.getWorkGroupSize() << std::endl;
		std::cout << "StoreKernel WorkGroupSize: " << tripletThetaPhiStore.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "TripletThetaPhiFilter_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "TripletThetaPhiFilter_STORE";}

	TrackletCollection * run(HitCollection & hits, const DetectorGeometry & geom, const GeometrySupplement & geomSupplement, const Dictionary & dict,
			int nThreads, int layers[], const LayerSupplement & layerSupplement, const Grid & grid,
			float dThetaCut, float dThetaWindow, float dPhiCut, float dPhiWindow, int pairSpreadZ)
	{

		//clever::vector<uint2,1> * m_pairs = generateAllPairs(hits, nThreads, layers, layerSupplement);
		PairGeneratorSector pairGen(ctx);
		clever::vector<uint2,1> * m_pairs = pairGen.run(hits, nThreads, layers, layerSupplement , grid, pairSpreadZ);

		//clever::vector<uint2,1> * m_triplets = generateAllTriplets(hits, nThreads, layers, hitCount, 1.2*dThetaCut, 1.2*dPhiCut, *m_pairs);
		TripletThetaPhiPredictor predictor(ctx);
		clever::vector<uint2,1> * m_triplets = predictor.run(hits, geom, geomSupplement, dict, nThreads, layers, layerSupplement, grid, dThetaWindow, dPhiWindow, *m_pairs);
		int nTripletCandidates = m_triplets->get_count();

		std::cout << "Initializing oracle...";
		clever::vector<uint, 1> m_oracle(0, std::ceil(nTripletCandidates / 32.0), ctx);
		std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

		std::cout << "Initializing prefix sum...";
		clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
		std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		std::cout << "Running filter kernel...";
		cl_event evt = tripletThetaPhiCheck.run(
				//configuration
				dThetaCut, dPhiCut,
				nTripletCandidates,
				// input
				m_pairs->get_mem(), m_triplets->get_mem(),
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
				// output
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum...";
		std::vector<uint> prefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
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
		int nFoundTriplets = prefixSum.run(m_prefixSum, nThreads, true);

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum...";
		std::vector<uint> prefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

		TrackletCollection * tracklets = new TrackletCollection(nFoundTriplets);
		std::cout << "Reserving space for " << nFoundTriplets << " tracklets" << std::endl;

		tracklets->transfer.initBuffers(ctx, *tracklets);

		std::cout << "Running filter store kernel...";
		evt = tripletThetaPhiStore.run(
				//configuration
				nTripletCandidates,
				//input
				m_pairs->get_mem(), m_triplets->get_mem(),
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//output
				// output
				tracklets->transfer.buffer(TrackletHit1()), tracklets->transfer.buffer(TrackletHit2()), tracklets->transfer.buffer(TrackletHit3()),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_STORE_EVT());

		std::cout << "Fetching triplets...";
		tracklets->transfer.fromDevice(ctx, *tracklets);
		std::cout <<"done[" << tracklets->size() << "]" << std::endl;

		//clean up
		delete m_pairs;
		delete m_triplets;

		return tracklets;
	}

	clever::vector<uint2,1> * generateAllPairs(HitCollection & hits, int nThreads, int layers[], const LayerSupplement & layerSupplement) {

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

	clever::vector<uint2,1> * generateAllTriplets(HitCollection & hits, int nThreads, int layers[], int hitCount[],
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

	KERNEL10_CLASS( tripletThetaPhiCheck, float, float, uint, cl_mem, cl_mem,  cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
			__kernel void tripletThetaPhiCheck(
					//configuration
					float dThetaCut, float dPhiCut, uint nTriplets,
					// hit input
					__global const uint2 * pairs, __global const uint2 * triplets,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global uint * prefixSum )
	{
		const size_t gid = get_global_id( 0 );
		const size_t lid = get_local_id( 0 );
		const size_t threads = get_global_size( 0 );

		uint workload = nTriplets / threads + 1;
		uint i = gid * workload;
		uint end = min(i + workload, nTriplets); // for last thread, if not a full workload is present
		uint nValid = 0;

		//printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nTriplets);

		for(; i < end; ++i){

			uint firstHit = pairs[triplets[i].x].x;
			uint secondHit = pairs[triplets[i].x].y;
			uint thirdHit = triplets[i].y;
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
			nValid += + valid;

			//if(valid)
			//	printf("[ %lu ] Found valid track %i (%i-%i-%i). Word %i Bit %i\n", id, i, firstHit, secondHit, thirdHit, i / 32, i % 32);

			atomic_or(&oracle[i / 32], (valid << (i % 32)));

		} // end triplet candidate loop

		prefixSum[gid] = nValid;
	});

	KERNEL8_CLASS( tripletThetaPhiStore, uint, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
				__kernel void tripletThetaPhiStore(
						//configuration
						uint nTriplets,
						// hit input
						__global const uint2 * pairs, __global const uint2 * triplets,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output of tracklet data
						__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3)
		{
			size_t id = get_global_id( 0 );
			size_t threads = get_global_size( 0 );

			uint workload = nTriplets / threads + 1;
			uint i = id * workload;
			uint end = min(i + workload, nTriplets); // for last thread, if not a full workload is present

			uint pos = prefixSum[id];

			for(; i < end && pos < prefixSum[id+1]; ++i){ // pos < prefixSum[id+1] can lead to thread divergence

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
