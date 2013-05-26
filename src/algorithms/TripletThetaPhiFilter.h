#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>

#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Pairings.h>
#include <datastructures/TripletConfiguration.h>
#include <datastructures/Grid.h>
#include <datastructures/Logger.h>

#include <datastructures/KernelWrapper.hpp>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class TripletThetaPhiFilter: public KernelWrapper<TripletThetaPhiFilter>
{

public:

	TripletThetaPhiFilter(clever::context & ctext) :
		KernelWrapper(ctext),
		filterCount(ctext),
		filterStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "FilterKernel WorkGroupSize: " << filterCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << filterStore.getWorkGroupSize() << std::endl;
}

	TrackletCollection * run(HitCollection & hits, const Grid & grid,
			const Pairing & pairs, const Pairing & tripletCandidates,
			int nThreads, const TripletConfigurations & layerTriplets);

	KERNEL12_CLASSP( filterCount, cl_mem, cl_mem, cl_mem,
			cl_mem,
			cl_mem,  cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			oclDEFINES,
			__kernel void filterCount(
					//configuration
					__global const float * thetaCut, __global const float * phiCut, __global const float * maxTIP,
					// hit input
					__global const uint2 * pairs,
					__global const uint2 * triplets, __global const uint * hitTripletOffsets,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global const uint * oracleOffset, __global uint * prefixSum )
	{
		size_t thread = get_global_id(0); // thread
		size_t layerTriplet = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerTriplets = get_global_size(1); //total number of processed layer pairings

		uint offset = event*nLayerTriplets + layerTriplet;
		uint tripletOffset = hitTripletOffsets[offset]; //offset of hit pairs
		uint i = tripletOffset + thread;
		uint end = hitTripletOffsets[offset + 1]; //last hit pair

		PRINTF(("%lu-%lu-%lu: from triplet candidate %u to %u\n", event, layerTriplet, thread, i, end));

		uint oOffset = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
		uint nValid = 0;

		float dThetaCut = thetaCut[layerTriplet];
		float dPhiCut = phiCut[layerTriplet];
		float tipCut = maxTIP[layerTriplet];

		PRINTF(("%lu-%lu-%lu: oracle from %u\n", event, layerTriplet, thread, oOffset));

		//PRINTF("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nTriplets);

		for(; i < end; i += threads){

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
			float delta = fabs(angle2/angle1);
			valid = valid * (1-dThetaCut <= delta && delta <= 1+dThetaCut);

			//tanPhi1
			angle1 = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) , ( hitGlobalX[secondHit] - hitGlobalX[firstHit] ));
			//tanPhi2
			angle2 = atan2((hitGlobalY[thirdHit] - hitGlobalY[secondHit]) , ( hitGlobalX[thirdHit] - hitGlobalX[secondHit] ));

			delta = angle2 - angle1;
			delta += (delta>180) ? -360 : (delta<-180) ? 360 : 0; //fix wrap around
			valid = valid * (fabs(delta) <= dPhiCut);

			//circle fit
			//map points to parabloid: (x,y) -> (x,y,x^2+y^2)
			float3 pP1 = (float3) (hitGlobalX[firstHit],
					hitGlobalY[firstHit],
					hitGlobalX[firstHit] * hitGlobalX[firstHit] + hitGlobalY[firstHit] * hitGlobalY[firstHit]);

			float3 pP2 = (float3) (hitGlobalX[secondHit],
					hitGlobalY[secondHit],
					hitGlobalX[secondHit] * hitGlobalX[secondHit] + hitGlobalY[secondHit] * hitGlobalY[secondHit]);

			float3 pP3 = (float3) (hitGlobalX[thirdHit],
					hitGlobalY[thirdHit],
					hitGlobalX[thirdHit] * hitGlobalX[thirdHit] + hitGlobalY[thirdHit] * hitGlobalY[thirdHit]);

			//span two vectors
			float3 a = pP2 - pP1;
			float3 b = pP3 - pP1;

			//compute unit cross product
			float3 n = cross(a,b);
			n = normalize(n);

			//formula for orign and radius of circle from Strandlie et al.
			float2 cOrigin = (float2) ((-n.x) / (2*n.z),
					(-n.y) / (2*n.z));

			float c = -(n.x*pP1.x + n.y*pP1.y + n.z*pP1.z);

			float cR = sqrt((1 - n.z*n.z - 4 * c * n.z) / (4*n.z*n.z));

			//find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
			float2 v = -cOrigin; v = normalize(v);
			float2 pCA = (float2) (cOrigin.x + cR*v.x,
					cOrigin.y + cR*v.y);

			//TIP = distance of point of closest approach to origin
			float tip = sqrt(pCA.x*pCA.x + pCA.y*pCA.y);

			valid = valid * (tip <= tipCut);

			//found good triplet?
			nValid += valid;

			uint index = (i - tripletOffset);

			PRINTF((valid ? "%lu-%lu-%lu: setting bit for %u -> %u\n" : "", event, layerTriplet, thread, i,  index));

			atomic_or(&oracle[(oOffset + index) / 32], (valid << (index % 32)));
			//oracle[i / 32] |= (valid << (i % 32));

		} // end triplet candidate loop

		prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread] = nValid;
	});

	KERNEL10_CLASSP( filterStore, cl_mem,
			cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem,
			oclDEFINES,
				__kernel void filterStore(
						// hit input
						__global const uint2 * pairs,
						__global const uint2 * triplets, __global const uint * hitTripletOffsets,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * oracleOffset,__global const uint * prefixSum,
						// output of tracklet data
						__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3,
						__global uint * trackletOffsets )
		{
		size_t thread = get_global_id(0); // thread
		size_t layerTriplet = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerTriplets = get_global_size(1); //total number of processed layer pairings

		uint offset = event*nLayerTriplets + layerTriplet;
		uint tripletOffset = hitTripletOffsets[offset]; //offset of hit pairs
		uint i = tripletOffset + thread;
		uint end = hitTripletOffsets[offset + 1]; //last hit pair

		PRINTF(("%lu-%lu-%lu: from triplet candidate %u to %u\n", event, layerTriplet, thread, i, end));

		uint pos = prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread]; //first position to write
		uint nextThread = prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread+1]; //first position of next thread

		PRINTF(("%lu-%lu-%lu: second layer from %u\n", event, layerTriplet, thread, offset));

		//configure oracle
		uint byte = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
		//uint bit = (byte + i*nHits2) % 32;
		//byte += (i*nHits2); byte /= 32;
		//uint sOracle = oracle[byte];

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u using memory %u to %u\n", event, layerTriplet, thread, i, end, pos, nextThread));


			for(; i < end && pos < nextThread; i += threads){ // pos < prefixSum[id+1] can lead to thread divergence

				//is this a valid triplet?
				uint index = i-tripletOffset;
				bool valid = oracle[(byte + index) / 32] & (1 << (index % 32));

				PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u -> %u written at %u\n" : "", event, layerTriplet, thread, i,  index, pos));


				//performance gain?
				/*bool valid = sOracle & (1 << bit);
				++bit;
				if(bit == 32){ //no divergence, all threads enter branch at the same time
					bit = 0;
					++byte;
					sOracle=oracle[byte];
				}*/

				//last triplet written on [pos] is valid one
				if(valid){
					trackletHitId1[pos] = pairs[triplets[i].x].x;
					trackletHitId2[pos] = pairs[triplets[i].x].y;
					trackletHitId3[pos] = triplets[i].y;
				}

				//if(valid)
				//	PRINTF("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

				//advance pos if valid
				pos += valid;
			}

		if(thread == threads-1){ //store pos in pairOffset array
				trackletOffsets[event * nLayerTriplets + layerTriplet + 1] = nextThread;
		}
		});

};
