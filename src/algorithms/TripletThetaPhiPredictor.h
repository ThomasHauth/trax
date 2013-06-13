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
class TripletThetaPhiPredictor: public KernelWrapper<TripletThetaPhiPredictor>
{

public:

	TripletThetaPhiPredictor(clever::context & ctext) :
		KernelWrapper(ctext),
		predictCount(ctext),
		predictStore(ctext),
		predictOffsetMonotonizeStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "PredictKernel WorkGroupSize: " << predictCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << predictStore.getWorkGroupSize() << std::endl;
}

	Pairing * run(HitCollection & hits, const DetectorGeometry & geom, const GeometrySupplement & geomSupplement, const Dictionary & dict,
			int nThreads, const TripletConfigurations & layerTriplets, const Grid & grid, const Pairing & pairs);

	KERNEL26_CLASSP( predictCount, cl_mem, cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, uint,
			cl_float, cl_float, uint,
			cl_float, cl_float, uint,
			cl_mem, cl_mem, cl_float,
			cl_mem, cl_uint,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			cl_mem, cl_mem,
			cl_mem,
			oclDEFINES,// "#define PRINTF(a) printf a",

	__kernel void predictCount(
					//detector geometry
					__global const uchar * detRadius, __global const float * radiusDict,__global const float * gMinLayerRadius, __global const float * gMaxLayerRadius,
					//grid data structure
					__global const uint * grid, __global const uint * layer3, const uint nLayers,
					const float minZ, const float sectorSizeZ, const uint nSectorsZ,
					const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
					//configuration
					__global const float * thetaWindow, __global const float * phiWindow, const float minRadiusCurvature,
					// hit input
					__global const uint2 * pairs, const uint nPairs,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global const uint * hitEvent, __global const uint * hitLayer,
					__global const uint * detId, __global const int * hitId,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * prefixSum)
	{

		size_t gid = get_global_id(0); // thread

		PRINTF(("%lu:\n", gid));

		if(gid < nPairs){ //divergence only in last work group
			//get hit pair
			uint firstHit = pairs[gid].x;
			uint secondHit = pairs[gid].y;

			PRINTF(("%lu: pair %u-%u\n", gid, firstHit, secondHit));

			uint event = hitEvent[firstHit]; // must be the same as hitEvent[secondHit] --> ensured during pair building
			uint layerTriplet = hitLayer[firstHit] - 1; //the layerTriplet is defined by its innermost layer --> TODO modify storage of layer triplets

			uint layer = layer3[layerTriplet]-1; //outer layer
			__global const uint * lGrid3 = &grid[event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)]; //offset into grid for easier access

			PRINTF(("%lu: pair %u-%u of event %u, layerTriplet %u outer layer: %u\n", gid, firstHit, secondHit, event, layerTriplet, layer));

			//PRINTF(("%lu-%lu-%lu: second layer from %u with %u hits\n", event, layerTriplet, thread, offset, nHits3));

			//ulong oOffset = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
			uint nFound = 0;

			float dThetaWindow = thetaWindow[layerTriplet];
			float dPhiWindow = phiWindow[layerTriplet];
			float minLayerRadius = gMinLayerRadius[layer];
			float maxLayerRadius = gMaxLayerRadius[layer];


			PRINTF(("id %lu loaded configuration\n", gid));

			//load hit data
			float3 p1 =  (float3) (hitGlobalX[firstHit], hitGlobalY[firstHit], hitGlobalZ[firstHit]);

			float3 p2 =  (float3) (hitGlobalX[secondHit], hitGlobalY[secondHit], hitGlobalZ[secondHit]);

			//calculate transverse impact parameter of straight line p1 -> p2
			// tip2 = x2 + f(x)2
			// with f(x) = m * x + n
			// calculate m and n with p1 and p2
			// diverentiate tip2 -> set zero -> obtain x0 -> plug it in and get minimal tip

			float tip2 = p1.y * (p2.x - p1.x) - p1.x * (p2.y - p1.y);
			tip2 *= tip2;
			tip2 /= ( (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y) ); //short form to get r min

			//the origin is p1
			// the cotangent is defined by p1 and p2 minus tip
			float rOrigin = sqrt(p1.x*p1.x + p1.y*p1.y);
			rOrigin = sqrt(rOrigin*rOrigin - tip2);
			float cotTheta = (p2.z - p1.z) / (sqrt(p2.x*p2.x + p2.y*p2.y - tip2) - rOrigin);


			//predict z with minimum and maximum layer radisu
			float tmp = p1.z + (sqrt(maxLayerRadius*maxLayerRadius - tip2)-rOrigin)*cotTheta;
			float zLow = p1.z + (sqrt(minLayerRadius*minLayerRadius - tip2)-rOrigin)*cotTheta;

			float zHigh = (tmp > zLow) * tmp + (tmp < zLow) * zLow;
			zLow = (tmp > zLow) * zLow + (tmp < zLow)*tmp;

			zLow -= 0.037;
			zHigh += 0.037;

			uint zLowSector = max((int) floor((zLow - minZ) / sectorSizeZ), 0); // signed int because zLow could be lower than minZ
			uint zHighSector = min((uint) floor((zHigh - minZ) / sectorSizeZ)+1, nSectorsZ);

			PRINTF(("%u-%u-%lu: hit2 %u -> prediction %f-%f [%u,%u]\n", event, layerTriplet, gid, secondHit, zLow, zHigh, zLowSector, zHighSector));

			//phi
			float phi = atan2(hitGlobalY[secondHit], hitGlobalX[secondHit]); //phi second hit
			tmp = atan2(hitGlobalY[firstHit], hitGlobalX[firstHit]); //phi first hit

			float dPhi = phi - tmp; //delta might be ]-pi,pi[
			dPhi += (dPhi>M_PI_F) ? -2*M_PI_F : (dPhi<-M_PI_F) ? 2*M_PI_F : 0; //fix wrap around
			dPhi = fabs(dPhi); //absolute value

			//printf("D1: %f\n", dPhi);

			float dHits = sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit]) * (hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit]) * (hitGlobalY[secondHit] - hitGlobalY[firstHit]));

			tmp = fabs(acos(dHits / (2 * minRadiusCurvature)) - acos(maxLayerRadius / (2 * minRadiusCurvature)));
			//use phi low as tempory variable
			float phiLow = fabs(acos(dHits / (2 * minRadiusCurvature)) - acos(minLayerRadius / (2 * minRadiusCurvature)));
			dPhi = max(dPhi, max(tmp, phiLow));

			//printf("D2: %f\n", dPhi);

			float phiHigh = phi + dPhi; // phi high may be greater than PI
			phiLow = phi - dPhi;

			//deal with wrap around
			bool wrapAround = phiLow < -M_PI_F || phiHigh > M_PI_F || phiLow > M_PI_F || phiHigh < -M_PI_F;

			phiLow -= (phiLow > M_PI_F) * 2 * M_PI_F;
			phiLow += (phiLow < -M_PI_F) * 2 * M_PI_F;

			phiHigh -= (phiHigh > M_PI_F) * 2 * M_PI_F;
			phiHigh += (phiHigh < -M_PI_F) * 2 * M_PI_F;

			uint phiLowSector= max((uint) floor((phiLow - minPhi) / sectorSizePhi), 0u);
			uint phiHighSector = min((uint) floor((phiHigh - minPhi) / sectorSizePhi)+1, nSectorsPhi);

			PRINTF(("%u-%u-%lu: hit2 %u:  phi = %f - %f -> [%u,%u] %s\n", event, layerTriplet, gid, secondHit, phiLow, phiHigh, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));

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
					ulong index = j - (j >= zSectorEnd) * zSectorLength;
					bool valid = zLow <= hitGlobalZ[index] && hitGlobalZ[index] <= zHigh;

					/*
					if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[index]){
						PRINTF("%i-%i-%i [%i]: z exp[%f]: %f - %f; z act: %f\n", firstHit, secondHit, index, hitId[firstHit], hitGlobalZ[secondHit], zLow, zHigh, hitGlobalZ[index]);
						float thetaAct = atan2(sign(hitGlobalY[index])*sqrt((hitGlobalX[index] - hitGlobalX[secondHit])*(hitGlobalX[index] - hitGlobalX[secondHit])
								+ (hitGlobalY[index] - hitGlobalY[secondHit])*(hitGlobalY[index] - hitGlobalY[secondHit]))
								, ( hitGlobalZ[index] - hitGlobalZ[secondHit] ));
						//if(!(thetaLow <= thetaAct && thetaAct <= thetaHigh))
						PRINTF("\ttheta exp[%f]: %f - %f; theta act: %f\n", theta, atan(1/cotThetaLow), atan(1/cotThetaHigh), thetaAct);
						//else {
						float r2 = sqrt(hitGlobalX[secondHit]*hitGlobalX[secondHit] + hitGlobalY[secondHit]*hitGlobalY[secondHit]);
						float r3 = sqrt(hitGlobalX[index]*hitGlobalX[index] + hitGlobalY[index]*hitGlobalY[index]);

						PRINTF("\tdr exp: %f - %f; dr act: %f\n", dRmin, dRmax, r3-r2);
						//}
					}
					 */

					// check phi range
					float actPhi = atan2(hitGlobalY[index],hitGlobalX[index]);
					valid = valid * ((!wrapAround && (phiLow <= actPhi && actPhi <= phiHigh))
							|| (wrapAround && ((phiLow <= actPhi && actPhi <= M_PI_F) || (-M_PI_F <= actPhi && actPhi <= phiHigh))));

					/*
					if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[index]){
						PRINTF("%i-%i-%i [%i]: phi exp: %f - %f; phi act: %f\n", firstHit, secondHit, index, hitId[firstHit], phiLow, phiHigh, actPhi);
						//}
					}
					 */

					//if valid update nFound
					nFound = nFound + valid;

					//update oracle
					//index = (i - pairOffset)*nHits3 + j - (j >= zSectorEnd) * zSectorLength - offset;

					//PRINTF((valid ? "%lu-%lu-%lu: setting bit for %u and %u (%u)\n" : "", event, layerTriplet, thread, i-pairOffset, j - (j >= zSectorEnd) * zSectorLength, j));

					//if(valid && ((oOffset + index) / 32) > 89766864)
					//printf("%lu-%lu-%lu: setting bit for %u and %u (%u) -> %u: %u-%u\n", event, layerTriplet, thread, i-pairOffset, j - (j >= zSectorEnd) * zSectorLength, j,  index, ((oOffset + index) / 32), (valid << (index % 32)));

					//atomic_or(&oracle[(oOffset + index) / 32], (valid << (index % 32)));

				} // end hit loop
			} // end sector loop

			prefixSum[gid] = nFound;

			//PRINTF("[%lu] rejZ: %u, rejP: %u, rejB: %u\n", gid, rejZ, rejP, rejB);
		}
	});

	KERNEL28_CLASSP(predictStore, cl_mem, cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, uint,
			cl_float, cl_float, uint,
			cl_float, cl_float, uint,
			cl_mem, cl_mem, cl_uint, cl_float,
			cl_mem, cl_uint,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			cl_mem,
			cl_mem,
			cl_mem, cl_mem,
			oclDEFINES,//"#define PRINTF(a) printf a",

	__kernel void predictStore(
					//detector geometry
					__global const uchar * detRadius, __global const float * radiusDict,__global const float * gMinLayerRadius, __global const float * gMaxLayerRadius,
					//grid data structure
					__global const uint * grid, __global const uint * layer3, const uint nLayers,
					const float minZ, const float sectorSizeZ, const uint nSectorsZ,
					const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
					// configuration
					__global const float * thetaWindow, __global const float * phiWindow, const uint nLayerTriplets, const float minRadiusCurvature,
					// hit input
					__global const uint2 * pairs, const uint nPairs,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global const uint * hitEvent, __global const uint * hitLayer,
					__global const uint * detId,
					// input for oracle and prefix sum
					__global const uint * prefixSum,
					// output triplet candidates
					__global uint2 * triplets, __global uint * tripletOffsets)
	{
		size_t gid = get_global_id(0); // thread

		PRINTF(("%lu:\n", gid));

		if(gid < nPairs){
			//get hit pair
			uint firstHit = pairs[gid].x;
			uint secondHit = pairs[gid].y;

			PRINTF(("%lu: pair %u-%u\n", gid, firstHit, secondHit));

			uint event = hitEvent[firstHit]; // must be the same as hitEvent[secondHit] --> ensured during pair building
			uint layerTriplet = hitLayer[firstHit]-1; //the layerTriplet is defined by its innermost layer --> TODO modify storage of layer triplets

			uint layer = layer3[layerTriplet]-1; //outer layer
			__global const uint * lGrid3 = &grid[event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)]; //offset into grid for easier access

			uint pos = prefixSum[gid]; //first position to write
			uint nextThread = prefixSum[gid+1]; //first position of next thread

			float dThetaWindow = thetaWindow[layerTriplet];
			float dPhiWindow = phiWindow[layerTriplet];
			float minLayerRadius = gMinLayerRadius[layer];
			float maxLayerRadius = gMaxLayerRadius[layer];

			PRINTF(("%lu: pair %u-%u of event %u, layerTriplet %u\n", gid, firstHit, secondHit, event, layerTriplet));

			//configure oracle
			//ulong byte = oracleOffset[event*nLayerTriplets+layerTriplet]; //offset in oracle array
			//uint bit = (byte + i*nHits2) % 32;
			//byte += (i*nHits2); byte /= 32;
			//uint sOracle = oracle[byte];

			//(("%lu-%lu-%lu: from hit1 %u to %u with hits2 %u using memory %u to %u\n", event, layerTriplet, thread, i, end, nHits3, pos, nextThread));
			//for(; i < end; i += threads){

			//theta
			//load hit data
			float3 p1 =  (float3) (hitGlobalX[firstHit], hitGlobalY[firstHit], hitGlobalZ[firstHit]);

			float3 p2 =  (float3) (hitGlobalX[secondHit], hitGlobalY[secondHit], hitGlobalZ[secondHit]);

			//calculate transverse impact parameter of straight line p1 -> p2
			// tip2 = x2 + f(x)2
			// with f(x) = m * x + n
			// calculate m and n with p1 and p2
			// diverentiate tip2 -> set zero -> obtain x0 -> plug it in and get minimal tip

			float tip2 = p1.y * (p2.x - p1.x) - p1.x * (p2.y - p1.y);
			tip2 *= tip2;
			tip2 /= ( (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y) ); //short form to get r min

			//the origin is p1
			// the cotangent is defined by p1 and p2 minus tip
			float rOrigin = sqrt(p1.x*p1.x + p1.y*p1.y);
			rOrigin = sqrt(rOrigin*rOrigin - tip2);
			float cotTheta = (p2.z - p1.z) / (sqrt(p2.x*p2.x + p2.y*p2.y - tip2) - rOrigin);


			//predict z with minimum and maximum layer radisu
			float tmp = p1.z + (sqrt(maxLayerRadius*maxLayerRadius - tip2)-rOrigin)*cotTheta;
			float zLow = p1.z + (sqrt(minLayerRadius*minLayerRadius - tip2)-rOrigin)*cotTheta;

			float zHigh = (tmp > zLow) * tmp + (tmp < zLow) * zLow;
			zLow = (tmp > zLow) * zLow + (tmp < zLow)*tmp;

			zLow -= 0.037;
			zHigh += 0.037;

			uint zLowSector = max((int) floor((zLow - minZ) / sectorSizeZ), 0); // signed int because zLow could be lower than minZ
			uint zHighSector = min((uint) floor((zHigh - minZ) / sectorSizeZ)+1, nSectorsZ);

			//PRINTF(("%lu-%lu-%lu: hit pair %u -> prediction %f-%f [%u,%u]\n", event, layerTriplet, thread, i, zLow, zHigh, zLowSector, zHighSector));

			//phi
			float phi = atan2(hitGlobalY[secondHit], hitGlobalX[secondHit]); //phi second hit
			tmp = atan2(hitGlobalY[firstHit], hitGlobalX[firstHit]); //phi first hit

			float dPhi = phi - tmp; //delta might be ]-pi,pi[
			dPhi += (dPhi>M_PI_F) ? -2*M_PI_F : (dPhi<-M_PI_F) ? 2*M_PI_F : 0; //fix wrap around
			dPhi = fabs(dPhi); //absolute value

			float dHits = sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit]) * (hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit]) * (hitGlobalY[secondHit] - hitGlobalY[firstHit]));

			tmp = fabs(acos(dHits / (2 * minRadiusCurvature)) - acos(maxLayerRadius / (2 * minRadiusCurvature)));
			//use phi low as tempory variable
			float phiLow = fabs(acos(dHits / (2 * minRadiusCurvature)) - acos(minLayerRadius / (2 * minRadiusCurvature)));
			dPhi = max(dPhi, max(tmp, phiLow));

			float phiHigh = phi + dPhi; // phi high may be greater than PI
			phiLow = phi - dPhi;

			//deal with wrap around
			bool wrapAround = phiLow < -M_PI_F || phiHigh > M_PI_F || phiLow > M_PI_F || phiHigh < -M_PI_F;

			phiLow -= (phiLow > M_PI_F) * 2 * M_PI_F;
			phiLow += (phiLow < -M_PI_F) * 2 * M_PI_F;

			phiHigh -= (phiHigh > M_PI_F) * 2 * M_PI_F;
			phiHigh += (phiHigh < -M_PI_F) * 2 * M_PI_F;

			uint phiLowSector= max((uint) floor((phiLow - minPhi) / sectorSizePhi), 0u);
			uint phiHighSector = min((uint) floor((phiHigh - minPhi) / sectorSizePhi)+1, nSectorsPhi);

			//PRINTF(("%lu-%lu-%lu: hit pair %u:  phi = %f - %f -> [%u,%u] %s\n", event, layerTriplet, thread, i, phiLow, phiHigh, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));

			for(uint zSector = zLowSector; zSector < zHighSector; ++ zSector){

				uint zSectorStart = lGrid3[(zSector)*(nSectorsPhi+1)];
				uint zSectorEnd = lGrid3[(zSector+1)*(nSectorsPhi+1)];
				uint zSectorLength = zSectorEnd - zSectorStart;

				uint j = lGrid3[zSector*(nSectorsPhi+1)+phiLowSector];
				uint end2 = wrapAround * zSectorEnd + //add end of layer
						lGrid3[zSector*(nSectorsPhi+1)+phiHighSector] //actual end of sector
						       - wrapAround * (zSectorStart); //substract start of zSector

				for(; j < end2 && pos < nextThread; ++j){ // pos < prefixSum[id+1] can lead to thread divergence

					// check z range
					ulong index = j - (j >= zSectorEnd) * zSectorLength;
					bool valid = zLow <= hitGlobalZ[index] && hitGlobalZ[index] <= zHigh;

					/*
										if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[index]){
											PRINTF("%i-%i-%i [%i]: z exp[%f]: %f - %f; z act: %f\n", firstHit, secondHit, index, hitId[firstHit], hitGlobalZ[secondHit], zLow, zHigh, hitGlobalZ[index]);
											float thetaAct = atan2(sign(hitGlobalY[index])*sqrt((hitGlobalX[index] - hitGlobalX[secondHit])*(hitGlobalX[index] - hitGlobalX[secondHit])
													+ (hitGlobalY[index] - hitGlobalY[secondHit])*(hitGlobalY[index] - hitGlobalY[secondHit]))
													, ( hitGlobalZ[index] - hitGlobalZ[secondHit] ));
											//if(!(thetaLow <= thetaAct && thetaAct <= thetaHigh))
											PRINTF("\ttheta exp[%f]: %f - %f; theta act: %f\n", theta, atan(1/cotThetaLow), atan(1/cotThetaHigh), thetaAct);
											//else {
											float r2 = sqrt(hitGlobalX[secondHit]*hitGlobalX[secondHit] + hitGlobalY[secondHit]*hitGlobalY[secondHit]);
											float r3 = sqrt(hitGlobalX[index]*hitGlobalX[index] + hitGlobalY[index]*hitGlobalY[index]);

											PRINTF("\tdr exp: %f - %f; dr act: %f\n", dRmin, dRmax, r3-r2);
											//}
										}
					 */

					// check phi range
					float actPhi = atan2(hitGlobalY[index],hitGlobalX[index]);
					valid = valid * ((!wrapAround && (phiLow <= actPhi && actPhi <= phiHigh))
							|| (wrapAround && ((phiLow <= actPhi && actPhi <= M_PI_F) || (-M_PI_F <= actPhi && actPhi <= phiHigh))));

					/*
										if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[index]){
											PRINTF("%i-%i-%i [%i]: phi exp: %f - %f; phi act: %f\n", firstHit, secondHit, index, hitId[firstHit], phiLow, phiHigh, actPhi);
											//}
										}

					//is this a valid triplet?
					ulong index = (i - pairOffset)*nHits3 + j - (j >= zSectorEnd) * zSectorLength - offset;
					//bool valid = oracle[(byte + index) / 32] & (1 << (index % 32));

					PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u and %u -> %u written at %u\n" : "", event, layerTriplet, thread, i-pairOffset, offset+j,  index, pos));

					//performance gain?
					bool valid = sOracle & (1 << bit);
				++bit;
				if(bit == 32){
					bit = 0;
					++byte;
					sOracle=oracle[byte];
				}*/

					//last triplet written on [pos] is valid one
					if(valid){
						triplets[pos].x = gid;
						triplets[pos].y = index;
					}

					//if(valid)
					//	PRINTF("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

					//advance pos if valid
					pos += valid;
				}
			}
			//}

			//determine triplet offsets
			if(gid < nPairs-1){ //not the last hit pair
				uint nextHit = pairs[gid+1].x;
				uint nextEvent = hitEvent[nextHit]; // must be the same as hitEvent[secondHit] --> ensured during pair building
				uint nextLayerTriplet = hitLayer[nextHit]-1; //the layerTriplet is defined by its innermost layer --> TODO modify storage of layer triplets

				if(layerTriplet != nextLayerTriplet || event != nextEvent){ //this thread is the last one processing an element of this particular event and layer triplet
					tripletOffsets[event * nLayerTriplets + layerTriplet + 1] = nextThread;
				}
			} else {
				tripletOffsets[event * nLayerTriplets + layerTriplet + 1] = nextThread; // this is the last pair, just store it
			}
		}
	});

	KERNEL2_CLASSP( predictOffsetMonotonizeStore,
			cl_mem, cl_uint,
			oclDEFINES,

			__kernel void predictOffsetMonotonizeStore(
					__global uint * tripletOffsets, const uint nOffsets)
	{

		size_t gid = get_global_id(0);

		//printf("thread %lu\n", gid);
		if(0 < gid && gid <= nOffsets){
			if(tripletOffsets[gid] == 0)
				tripletOffsets[gid] = tripletOffsets[gid-1];
		}
	});

};
