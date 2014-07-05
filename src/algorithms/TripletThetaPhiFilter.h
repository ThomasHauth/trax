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
		filterPopCount(ctext),
		filterStore(ctext),
		filterOffsetStore(ctext),
		filterOffsetMonotonizeStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "FilterKernel WorkGroupSize: " << filterCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << filterStore.getWorkGroupSize() << std::endl;
}

	TrackletCollection * run(HitCollection & hits, const Grid & grid,
			const Pairing & pairs, const Pairing & tripletCandidates,
			int nThreads, const TripletConfigurations & layerTriplets);

	KERNEL_CLASSP( filterCount,
			oclDEFINES,
			__kernel void filterCount(
					//configuration
					__global const float * thetaCut, __global const float * phiCut, __global const float * maxTIP, const float minRadius,
					// hit input
					__global const uint2 * pairs,
					__global const uint2 * triplets, const uint nTriplets,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global const uint * hitEvent, __global const uint * hitLayer,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle )
	{
		size_t gid = get_global_id(0); // thread

		if(gid < nTriplets){
			//PRINTF(("%lu-%lu-%lu: from triplet candidate %u to %u\n", event, layerTriplet, thread, i, end));

			uint firstHit = pairs[triplets[gid].x].x;
			uint secondHit = pairs[triplets[gid].x].y;
			uint thirdHit = triplets[gid].y;

			uint event = hitEvent[firstHit]; // must be the same as hitEvent[secondHit] --> ensured during pair building
			uint layerTriplet = hitLayer[firstHit] - 1; //the layerTriplet is defined by its innermost layer

			float dThetaCut = thetaCut[layerTriplet];
			float dPhiCut = phiCut[layerTriplet];
			float tipCut = maxTIP[layerTriplet];

			//PRINTF(("%lu-%lu-%lu: oracle from %u\n", event, layerTriplet, thread, oOffset));

			//PRINTF("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nTriplets);

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
			delta += (delta>M_PI_F) ? -2*M_PI_F : (delta<-M_PI_F) ? 2*M_PI_F : 0; //fix wrap around
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

			//check for compliance with minimum radius, ie. min Pt

			valid = valid * (cR >= minRadius);

			//found good triplet?
			//nValid += valid;

			//uint index = (i - tripletOffset);

			//PRINTF((valid ? "%lu-%lu-%lu: setting bit for %u -> %u\n" : "", event, layerTriplet, thread, i,  index));

			atomic_or(&oracle[(gid) / 32], (valid << (gid % 32)));
			//oracle[i / 32] |= (valid << (i % 32));


			//prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread] = nValid;
		}
	},
		cl_mem, cl_mem, cl_mem, cl_float,
		cl_mem,
		cl_mem,  cl_uint,
		cl_mem, cl_mem, cl_mem,
		cl_mem, cl_mem,
		cl_mem);

	KERNEL_CLASSP(filterPopCount, oclDEFINES,

			__kernel void filterPopCount(__global const uint * oracle, __global uint * prefixSum, const uint n)
	{

		size_t gid = get_global_id(0);
		if(gid < n){
			// only present in OCL 1.2 prefixSum[gid] = popcount(oracle[gid]);
			uint i = oracle[gid];
			i = i - ((i >> 1) & 0x55555555);
			i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			prefixSum[gid] = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
		}

	}, cl_mem, cl_mem, cl_uint);

	KERNEL_CLASSP( filterStore,
			oclDEFINES,
				__kernel void filterStore(
						// hit input
						__global const uint2 * pairs, const uint nLayerTriplets,
						__global const uint2 * triplets, const uint nTriplets,
						// input for oracle and prefix sum
						__global const uint * oracle, const uint nOracleBytes,__global const uint * prefixSum,
						// output of tracklet data
						__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3,
						__global uint * trackletOffsets )
		{
		size_t gid = get_global_id(0); // thread

		if(gid < nOracleBytes){
			//PRINTF(("%lu-%lu-%lu: from triplet candidate %u to %u\n", event, layerTriplet, thread, i, end));

			uint pos = prefixSum[gid]; //first position to write
			uint nextThread = prefixSum[gid+1]; //first position of next thread

			//PRINTF(("%lu-%lu-%lu: second layer from %u\n", event, layerTriplet, thread, offset));

			//configure oracle
			uint lOracle = oracle[gid]; //load oracle byte
			//uint bit = (byte + i*nHits2) % 32;
			//byte += (i*nHits2); byte /= 32;
			//uint sOracle = oracle[byte];

			//PRINTF(("%lu-%lu-%lu: from hit1 %u to %u using memory %u to %u\n", event, layerTriplet, thread, i, end, pos, nextThread));

			//loop over bits of oracle byte
			for(uint i = 0; i < 32 /*&& pos < nextThread*/; ++i){ // pos < prefixSum[id+1] can lead to thread divergence

				//is this a valid triplet?
				bool valid = lOracle & (1 << i);

				//PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u -> %u written at %u\n" : "", event, layerTriplet, thread, i,  index, pos));


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
					trackletHitId1[pos] = pairs[triplets[gid*32 + i].x].x;
					trackletHitId2[pos] = pairs[triplets[gid*32 + i].x].y;
					trackletHitId3[pos] = triplets[gid*32 + i].y;
				}

				//if(valid)
				//	PRINTF("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

				//advance pos if valid
				pos += valid;
			}
		}
		}, cl_mem, cl_uint,
			cl_mem, cl_uint,
			cl_mem, cl_uint, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem);

	KERNEL_CLASSP( filterOffsetStore,
			oclDEFINES,

			__kernel void filterOffsetStore(
					//tracklets
					__global const uint * trackletHitId1, __global const uint * trackletHitId2, __global const uint * trackletHitId3,
					const uint nTracklets, const uint nLayerTriplets,
					//hit data
					__global const uint * hitEvent, __global const uint * hitLayer,
					//output tracklet offst
					__global uint * trackletOffsets)
	{

		size_t gid = get_global_id(0);

		//printf("thread %lu\n", gid);
		if(gid < nTracklets){
			// data for this tracklet
			uint event = trackletHitId1[gid]; //temp store for hit id
			uint layerTriplet = hitLayer[event]-1;
			event = hitEvent[event];

			//printf("thread %lu event %u layerTriplet %u\n", gid, event, layerTriplet);

			if(gid < nTracklets-1){
				// data for next tracklet
				uint nextEvent = trackletHitId1[gid+1]; //temp store for hit id
				uint nextLayerTriplet = hitLayer[nextEvent]-1;
				nextEvent = hitEvent[nextEvent];

				if(layerTriplet != nextLayerTriplet || event != nextEvent){ //this thread is the last one processing an element of this particular event and layer triplet
					trackletOffsets[event * nLayerTriplets + layerTriplet + 1] = gid+1;
					//printf("event %u layerTriplet %u: %lu\n", event, layerTriplet, gid+1);
				}
			} else {
				//printf("end thread %lu: writting %u at %u\n", gid, gid+1, event * nLayerTriplets + layerTriplet + 1);
				trackletOffsets[event * nLayerTriplets + layerTriplet + 1] = gid+1;
				//printf("event %u layerTriplet %u: %lu\n", event, layerTriplet, gid+1);
			}
		}
	},	cl_mem, cl_mem, cl_mem,
		cl_uint, cl_uint,
		cl_mem, cl_mem,
		cl_mem);

	KERNEL_CLASSP( filterOffsetMonotonizeStore,
			oclDEFINES,

			__kernel void filterOffsetMonotonizeStore(
					__global uint * trackletOffsets, const uint nOffsets)
	{

		size_t gid = get_global_id(0);

		//printf("thread %lu\n", gid);
		if(0 < gid && gid <= nOffsets){
			if(trackletOffsets[gid] == 0)
				trackletOffsets[gid] = trackletOffsets[gid-1];
		}
	}, cl_mem, cl_uint);



};
