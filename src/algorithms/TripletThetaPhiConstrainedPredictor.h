#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/TripletConfiguration.h>
#include <datastructures/Pairings.h>
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
class TripletThetaPhiConstrainedPredictor: public KernelWrapper<TripletThetaPhiConstrainedPredictor>
{

public:

	TripletThetaPhiConstrainedPredictor(clever::context & ctext) :
		KernelWrapper(ctext),
		predictCount(ctext),
		predictStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "PredictKernel WorkGroupSize: " << predictCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << predictStore.getWorkGroupSize() << std::endl;
}

	Pairing * run(HitCollection & hits, const DetectorGeometry & geom, const GeometrySupplement & geomSupplement, const Dictionary & dict,
			int nThreads, const TripletConfigurations & layerTriplets, const Grid & grid, const Pairing & pairs);

	KERNEL29_CLASSP( predictCount, cl_mem, cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, uint,
			cl_float, cl_float, uint,
			cl_float, cl_float, uint,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			local_param, local_param, local_param,
			oclDEFINES,


	void siftDown(__local float * values, __local int * indices, uint n) {

		uint root = 0;

		while((root<<1) +1 < n){ //has at least left child
			uint child = (root<<1) + 1; //left child
			uint swap = root;
			if(values[swap] < values[child])
				swap = child;
			if(child+1 < n && values[swap] < values[child+1]) //test whether right child exists and is smaller
				swap = child+1;
			if(swap != root){ //swap elements
				uint tI = indices[root];
				indices[root] = indices[swap];
				indices[swap] = tI;

				float tV = values[root];
				values[root] = values[swap];
				values[swap] = tV;
			} else //finished
				return;
		}

	}

	__kernel void predictCount(
					//detector geometry
					__global const uchar * detRadius, __global const float * radiusDict,__global const float * gMinLayerRadius, __global const float * gMaxLayerRadius,
					//grid data structure
					__global const uint * grid, __global const uint * layer3, const uint nLayers,
					const float minZ, const float sectorSizeZ, const uint nSectorsZ,
					const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
					//configuration
					__global const float * thetaWindow, __global const float * phiWindow, __global const uint * NCandidates,
					// hit input
					__global const uint2 * pairs, __global const uint * hitPairOffsets,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global const uint * detId, __global const int * hitId,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global const uint * oracleOffset, __global uint * prefixSum,
					//local buffers
					__local uint * lGrid3, __local int * sICandidate, __local float * sVCandidate)
	{

		size_t thread = get_global_id(0); // thread
		size_t layerTriplet = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerTriplets = get_global_size(1); //total number of processed layer pairings

		uint offset = event*nLayerTriplets + layerTriplet;
		uint pairOffset = hitPairOffsets[offset]; //offset of hit pairs
		uint i = pairOffset + thread;
		uint end = hitPairOffsets[offset + 1]; //last hit pair

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u\n", event, layerTriplet, thread, i, end));

		uint layer = layer3[layerTriplet]-1; //outer layer
		uint nHits3 = (nSectorsZ+1)*(nSectorsPhi+1); //temp: number of grid cells
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in grid data structure for outer layer
		//load grid for second layer to local mem
		for(uint i = thread; i < nHits3; i += threads){
			lGrid3[i] = grid[offset + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		nHits3 = lGrid3[nHits3-1] - lGrid3[0]; //number of hits in outer layer
		offset = lGrid3[0]; //beginning of outer layer

		PRINTF(("%lu-%lu-%lu: second layer from %u with %u hits\n", event, layerTriplet, thread, offset, nHits3));

		uint oOffset = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
		uint nFound = 0;

		float dThetaWindow = thetaWindow[layerTriplet];
		float dPhiWindow = phiWindow[layerTriplet];
		float minLayerRadius = gMinLayerRadius[layer];
		float maxLayerRadius = gMaxLayerRadius[layer];
		uint nCandidates = NCandidates[layer];

		__local float * vCandidate = &sVCandidate[thread * nCandidates];
		__local int * iCandidate = &sICandidate[thread * nCandidates];

		PRINTF(("%lu-%lu-%lu: oracle from %u\n", event, layerTriplet, thread, oOffset));


		//PRINTF("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nPairs);

		for(; i < end; i += threads){ //workload loop

			uint firstHit = pairs[i].x;
			uint secondHit = pairs[i].y;

			//theta
			float signRadius = sign(hitGlobalY[secondHit]);
			float theta = atan2( signRadius * sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit])*(hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit])*(hitGlobalY[secondHit] - hitGlobalY[firstHit]))
																		, ( hitGlobalZ[secondHit] - hitGlobalZ[firstHit] ));

			float tmp = (1-dThetaWindow) * theta;
			int overflow = 1 - (fabs(tmp) > M_PI_F)*2;
			tmp -= (tmp > M_PI_F) * 2 * M_PI_F;
			tmp += (tmp < -M_PI_F) * 2 * M_PI_F;
			tmp *= overflow; //signRadius is set according to original theta --> it it overflows we must adjust angle to compensate for "wrongly" set signRadius
			float cotThetaLow = tan(M_PI_2_F - tmp);
			//int thetaLowSgn = 1 - (fabs(thetaLow) > M_PI_2_F) * 2;
			//thetaLow = (fabs(thetaLow) <= M_PI_2_F) * thetaLow + (fabs(thetaLow) > M_PI_2_F) * (sign(thetaLow)*M_PI_F - thetaLow);

			tmp = (1+dThetaWindow) * theta;
			overflow = 1 - (fabs(tmp) > M_PI_F)*2;
			tmp -= (tmp > M_PI_F) * 2 * M_PI_F;
			tmp += (tmp < -M_PI_F) * 2 * M_PI_F;
			tmp *= overflow;
			float cotThetaHigh = tan(M_PI_2_F -tmp);
			//int thetaHighSgn = 1 - (fabs(thetaHigh) > M_PI_2_F) * 2;
			//thetaHigh = (fabs(thetaHigh) <= M_PI_2_F) * thetaHigh + (fabs(thetaHigh) > M_PI_2_F) * (sign(thetaHigh)*M_PI_F - thetaHigh);

			//radius
			float r = signRadius * radiusDict[detRadius[detId[secondHit]]];
			float dRmax = signRadius * maxLayerRadius - r;
			float dRmin = signRadius * minLayerRadius - r;

			//z_3 = z_2 + dr * cot(theta) => cot(theta) = tan(pi/2 - theta)

			//first calculate for dRmax
			tmp = hitGlobalZ[secondHit] + dRmax * cotThetaLow;
			float zHigh = hitGlobalZ[secondHit] + dRmax * cotThetaHigh;

			float zLow = (tmp < zHigh) * tmp + (tmp > zHigh) * zHigh;
			zHigh = (tmp < zHigh) * zHigh + (tmp > zHigh) * tmp;

			//now for dRmin
			//thetaLow
			tmp = hitGlobalZ[secondHit] + dRmin * cotThetaLow;
			zLow = (tmp < zLow) * tmp + (tmp > zLow) * zLow;
			zHigh = (tmp > zHigh) * tmp + (tmp < zHigh) * zHigh;
			//thetaHigh
			tmp = hitGlobalZ[secondHit] + dRmin * cotThetaHigh;
			zLow = (tmp < zLow) * tmp + (tmp > zLow) * zLow;
			zHigh = (tmp > zHigh) * tmp + (tmp < zHigh) * zHigh;
			//now zLow - zHigh should be the maxiumum possible range

			uint zLowSector = max((int) floor((zLow - minZ) / sectorSizeZ), 0); // signed int because zLow could be lower than minZ
			uint zHighSector = min((uint) floor((zHigh - minZ) / sectorSizeZ)+1, nSectorsZ);

			PRINTF(("%lu-%lu-%lu: hit pair %u -> prediction %f-%f [%u,%u]\n", event, layerTriplet, thread, i, zLow, zHigh, zLowSector, zHighSector));

			//phi
			float phi = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) , ( hitGlobalX[secondHit] - hitGlobalX[firstHit] ));

			tmp = (1-dPhiWindow) * phi; //phi low may be smaller than -PI
			float phiHigh = (1+dPhiWindow) * phi; // phi high may be greater than PI
			float phiLow = (tmp < phiHigh) * tmp + (tmp > phiHigh) * phiHigh; //we sort before handling wrap around
			phiHigh = (tmp < phiHigh) * phiHigh + (tmp > phiHigh) * tmp; // if wrap around occurs, phiLow will be second quadrant and phiHigh will be third quadrant

			//deal with wrap around
			bool wrapAround = phiLow < -M_PI_F || phiHigh > M_PI_F || phiLow > M_PI_F || phiHigh < -M_PI_F;

			phiLow -= (phiLow > M_PI_F) * 2 * M_PI_F;
			phiLow += (phiLow < -M_PI_F) * 2 * M_PI_F;

			phiHigh -= (phiHigh > M_PI_F) * 2 * M_PI_F;
			phiHigh += (phiHigh < -M_PI_F) * 2 * M_PI_F;

			uint phiLowSector= max((uint) floor((phiLow - minPhi) / sectorSizePhi), 0u);
			uint phiHighSector = min((uint) floor((phiHigh - minPhi) / sectorSizePhi)+1, nSectorsPhi);

			PRINTF(("%lu-%lu-%lu: hit pair %u:  phi = %f - %f -> [%u,%u] %s\n", event, layerTriplet, thread, i, phiLow, phiHigh, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));

			for(uint cand = 0; cand < nCandidates; ++ cand){ //reset candidates list to zero
				vCandidate[cand] = MAXFLOAT;
				iCandidate[cand] = -1;
			}

			for(uint zSector = zLowSector; zSector < zHighSector; ++ zSector){

				uint zSectorStart = lGrid3[(zSector)*(nSectorsPhi+1)];
				uint zSectorEnd = lGrid3[(zSector+1)*(nSectorsPhi+1)];
				uint zSectorLength = zSectorEnd - zSectorStart;

				uint j = lGrid3[zSector*(nSectorsPhi+1)+phiLowSector];
				uint end2 = wrapAround * zSectorEnd + //add end of layer
						lGrid3[zSector*(nSectorsPhi+1)+phiHighSector] //actual end of sector
						                 - wrapAround * (zSectorStart); //substract start of zSector

				for(; j < end2; ++j){
					// check z range
					uint index = j - (j >= zSectorEnd) * zSectorLength;

					//calculate theta
					float theta2 = atan2( signRadius * sqrt((hitGlobalX[index] - hitGlobalX[secondHit])*(hitGlobalX[index] - hitGlobalX[secondHit])
										+ (hitGlobalY[index] - hitGlobalY[secondHit])*(hitGlobalY[index] - hitGlobalY[secondHit]))
																							, ( hitGlobalZ[index] - hitGlobalZ[secondHit] ));
					float phi2 = atan2((hitGlobalY[index] - hitGlobalY[secondHit]) , ( hitGlobalX[index] - hitGlobalX[secondHit] ));

					float value = phi2 - phi;
					value += (value>M_PI_2_F) ? -M_PI_F : (value<-M_PI_2_F) ? M_PI_F : 0; //fix wrap around
					value = fabs(value);
					//value now holds penalty for dPhi
					value += fabs(theta2/theta -1); //add penalty for dTheta

					if(value < vCandidate[0]){ //check whether current candidate is better than the worst found one
						vCandidate[0] = value;
						iCandidate[0] = (i - pairOffset)*nHits3 + j - (j >= zSectorEnd) * zSectorLength - offset; //index in oracle bitstring
						siftDown(vCandidate, iCandidate, nCandidates);
					}

				} // end hit loop
			} // end sector loop

			//all canidates inspected, set bits for best ones
			for(uint cand = 0; cand < nCandidates; ++ cand){
				int index = iCandidate[cand];

				if(index >= 0){
					//if valid update nFound
					++nFound;

					PRINTF((index >= 0 ? "%lu-%lu-%lu: setting bit for %u and %u\n" : "", event, layerTriplet, thread, i-pairOffset,  index));

					atomic_or(&oracle[(oOffset + index) / 32], (1 << (index % 32)));
				}
			}

		} //end workload loop

		prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread] = nFound;

		//PRINTF("[%lu] rejZ: %u, rejP: %u, rejB: %u\n", gid, rejZ, rejP, rejB);
	});

	KERNEL12_CLASSP( predictStore, cl_mem, uint, uint,
			cl_mem, uint,
			cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			oclDEFINES,
				__kernel void predictStore(
						//configuration
						__global const uint * grid, const uint nSectorsZ, const uint nSectorsPhi,
						__global const uint * layer3, const uint nLayers,
						// hit input
						__global const uint2 * pairs, __global const uint * hitPairOffsets,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * oracleOffset, __global const uint * prefixSum,
						// output triplet candidates
						__global uint2 * triplets, __global uint * tripletOffsets)
		{
		size_t thread = get_global_id(0); // thread
		size_t layerTriplet = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerTriplets = get_global_size(1); //total number of processed layer pairings

		uint offset = event*nLayerTriplets + layerTriplet;
		uint pairOffset = hitPairOffsets[offset]; //offset of hit pairs
		uint i = pairOffset + thread;
		uint end = hitPairOffsets[offset + 1]; //last hit pair

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u\n", event, layerTriplet, thread, i, end));

		uint layer = layer3[layerTriplet]-1; //outer layer
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint nHits3 = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1] - grid[offset]; //number of hits in second layer
		offset = grid[offset]; //beginning of outer layer

		uint pos = prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread]; //first position to write
		uint nextThread = prefixSum[event*nLayerTriplets*threads + layerTriplet*threads + thread+1]; //first position of next thread

		PRINTF(("%lu-%lu-%lu: second layer from %u with %u hits\n", event, layerTriplet, thread, offset, nHits3));

		//configure oracle
		uint byte = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
		//uint bit = (byte + i*nHits2) % 32;
		//byte += (i*nHits2); byte /= 32;
		//uint sOracle = oracle[byte];

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u with hits2 %u using memory %u to %u\n", event, layerTriplet, thread, i, end, nHits3, pos, nextThread));
		for(; i < end; i += threads){

			for(uint j = 0; j < nHits3 && pos < nextThread; ++j){ // pos < prefixSum[id+1] can lead to thread divergence

				//is this a valid triplet?
				uint index = (i - pairOffset)*nHits3+j;
				bool valid = oracle[(byte + index) / 32] & (1 << (index % 32));

				PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u and %u -> %u written at %u\n" : "", event, layerTriplet, thread, i-pairOffset, offset+j,  index, pos));

				//performance gain?
				/*bool valid = sOracle & (1 << bit);
				++bit;
				if(bit == 32){
					bit = 0;
					++byte;
					sOracle=oracle[byte];
				}*/

				//last triplet written on [pos] is valid one
				if(valid){
					triplets[pos].x = i;
					triplets[pos].y = offset + j;
				}

				//if(valid)
				//	PRINTF("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

				//advance pos if valid
				pos += valid;
			}
		}

		if(thread == threads-1){ //store pos in pairOffset array
			tripletOffsets[event * nLayerTriplets + layerTriplet + 1] = nextThread;
		}
		});

};