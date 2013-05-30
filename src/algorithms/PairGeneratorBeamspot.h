#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/TripletConfiguration.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Pairings.h>
#include <datastructures/Grid.h>
#include <datastructures/Logger.h>

#include <datastructures/KernelWrapper.hpp>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class PairGeneratorBeamspot: public KernelWrapper<PairGeneratorBeamspot>
{

public:

	PairGeneratorBeamspot(clever::context & ctext) :
		KernelWrapper(ctext),
		pairCount(ctext),
		pairStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "FilterKernel WorkGroupSize: " << pairCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << pairStore.getWorkGroupSize() << std::endl;
}

	Pairing * run(HitCollection & hits, const GeometrySupplement & geomSupplement,
				uint nThreads, const TripletConfigurations & layerTriplets, const Grid & grid);

	KERNEL22_CLASSP( pairCount, cl_mem, cl_mem,
			cl_mem, cl_mem, uint,
			cl_mem,
			cl_float, cl_float, uint,
			cl_float, cl_float, uint,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			local_param,
			oclDEFINES, //"#define PRINTF(a) printf a",
			__kernel void pairCount(
					//geometry
					__global const float * gMinLayerRadius, __global const float * gMaxLayerRadius,
					//grid
					__global const uint * layer1, __global const uint * layer2, const uint nLayers,
					__global const uint * grid,
					const float minZ, const float sectorSizeZ , const uint nSectorsZ,
					const float minPhi, const float sectorSizePhi , const uint nSectorsPhi,
					//configuration
					__global const float * gZ0, __global const uint * pairSpreadPhi, __global const float * thetaWindow,
					//hit data
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global uint * oracle, __global const uint * oracleOffset, __global uint * prefixSum,
					__local uint * lGrid1)
	{
		size_t thread = get_global_id(0); // thread
		size_t layerPair = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerPairs = get_global_size(1); //total number of processed layer pairings

		uint layer = layer2[layerPair]-1; //outer layer
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint layer2Offset = grid[offset]; //offset of outer layer
		uint i = layer2Offset + thread;
		uint end = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1]; //last hit of outer layer in hit array
		uint nHits2 = end - layer2Offset;

		PRINTF(("%lu-%lu-%lu: from hit2 %u to %u in layer 2 with %u hits\n", event, layerPair, thread, i, end, nHits2));

		layer = layer1[layerPair]-1; //inner layer
		uint nHits1 = (nSectorsZ+1)*(nSectorsPhi+1); //temp: number of grid cells
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in grid data structure for outer layer
		//load grid for second layer to local mem
		for(uint i = thread; i < nHits1; i += threads){
			lGrid1[i] = grid[offset + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		nHits1 = lGrid1[nHits1-1] - lGrid1[0]; //number of hits in first layer
		offset = lGrid1[0]; //beginning of inner layer

		PRINTF(("%lu-%lu-%lu: first layer from %u with %u hits\n", event, layerPair, thread, offset, nHits1));

		float z0 = gZ0[layerPair];
		uint spreadPhi = pairSpreadPhi[layerPair];

		float dTheta = thetaWindow[layerPair];
		float minLayerRadius = gMinLayerRadius[layer];
		float maxLayerRadius = gMaxLayerRadius[layer];

		PRINTF(("z0 %f, spread phi %u, dTheta %f layer %f-%f\n", z0, spreadPhi, dTheta, minLayerRadius, maxLayerRadius));

		uint oOffset = oracleOffset[event*nLayerPairs+layerPair]; //offset in oracle array
		uint nFound = 0;

		PRINTF(("%lu-%lu-%lu: oracle from %u\n", event, layerPair, thread, oOffset));

		//PRINTF(("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", gid, threads, workload, i, end, (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)])));

		for(; i < end; i += threads){ //loop over second layer

			//theta
			float signRadius = sign(hitGlobalY[i]);

			//calculate theta change due to LIP:  atan(radius / z +- z0) +- dTheta to account for effects -->
			float cotThetaHigh = atan2( signRadius * sqrt(hitGlobalX[i]*hitGlobalX[i] + hitGlobalY[i]*hitGlobalY[i]), hitGlobalZ[i] - z0) - signRadius * dTheta;
			cotThetaHigh = tan(M_PI_2_F - cotThetaHigh); //calculate cotangent

			float cotThetaLow = atan2( signRadius * sqrt(hitGlobalX[i]*hitGlobalX[i] + hitGlobalY[i]*hitGlobalY[i]), hitGlobalZ[i] + z0) + signRadius * dTheta;
			cotThetaLow = tan(M_PI_2_F - cotThetaLow);

			//first z high
			float tmp = signRadius * maxLayerRadius * cotThetaHigh + z0;
			float zHigh = signRadius * minLayerRadius * cotThetaHigh + z0;
			zHigh = (tmp < zHigh) * zHigh + (tmp > zHigh) * tmp;

			//now z low
			tmp = signRadius * maxLayerRadius * cotThetaLow - z0;
			float zLow = signRadius * minLayerRadius * cotThetaLow - z0;
			zLow = (tmp < zLow) * tmp + (tmp > zLow) * zLow;

			//calculate sectors
			uint zLowSector = max((int) floor((zLow - minZ) / sectorSizeZ), 0);
			uint zHighSector = min((uint) floor((zHigh - minZ) / sectorSizeZ)+1, nSectorsZ); // upper sector border equals sectorNumber + 1

			PRINTF(("%lu-%lu-%lu: hit2 %u:  z = %f r = %f -> %f - %f [%f,%f]\n", event, layerPair, thread, i, hitGlobalZ[i], signRadius * sqrt(hitGlobalX[i]*hitGlobalX[i] + hitGlobalY[i]*hitGlobalY[i]), zLow, zHigh, minZ+zLowSector*sectorSizeZ, minZ+zHighSector*sectorSizeZ));

			float phi = atan2(hitGlobalY[i], hitGlobalX[i]);

			int phiLowSector= floor((phi - minPhi) / sectorSizePhi) - spreadPhi; // lower phi sector, can underflow
			//PRINTF(("phi %f sector %f-%u\n", phi, (phi - minPhi) / sectorSizePhi, phiLowSector));
			uint phiHighSector = phiLowSector + 2*spreadPhi + 1; //higher phi sector, can overflow
			bool wrapAround = phiLowSector < 0 || phiHighSector > (nSectorsPhi + 1); // does wrap around occur?
			//PRINTF(("%lu-%lu-%lu: hit1 %u:  phi = %f -> [%i,%u] %s\n", event, layerPair, thread, i, phi, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));
			phiLowSector += (phiLowSector < 0) * (nSectorsPhi); //correct wraparound
			phiHighSector -= (phiHighSector > (nSectorsPhi + 1)) * (nSectorsPhi);

			PRINTF(("%lu-%lu-%lu: hit2 %u:  phi = %f -> [%i,%u] %s\n", event, layerPair, thread, i, phi, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));

			for(uint zSector = zLowSector; zSector < zHighSector; ++ zSector){

				uint zSectorStart = lGrid1[(zSector)*(nSectorsPhi+1)];
				uint zSectorEnd = lGrid1[(zSector+1)*(nSectorsPhi+1)-1];
				uint zSectorLength = zSectorEnd - zSectorStart;

				uint j = lGrid1[zSector*(nSectorsPhi+1)+phiLowSector];
				uint end2 = wrapAround * zSectorEnd + //add end of layer
						lGrid1[zSector*(nSectorsPhi+1)+phiHighSector] //actual end of sector
						                 - wrapAround * (zSectorStart); //substract start of zSector

				PRINTF(("%lu-%lu-%lu: hit1 from %u to %u\n", event, layerPair, thread, j, end2));

				for(; j < end2; ++j){

					++nFound;

					//update oracle
					//           skip to appropriate inner hit
					//								treat phi overflow
					//																beginning of second layer
					uint index = (j - (j >= zSectorEnd) * zSectorLength - offset)*nHits2 + i - layer2Offset;
					PRINTF(("%lu-%lu-%lu: setting bit for %u and %u -> %u\n", event, layerPair, thread, j - (j >= zSectorEnd) * zSectorLength - offset, i - layer2Offset, index));
					atomic_or(&oracle[(oOffset + index) / 32], (1 << (index % 32)));

				}

			} // end second hit loop

		} // end workload loop

		prefixSum[event*nLayerPairs*threads + layerPair*threads + thread] = nFound;

		//PRINTF("[%lu] Found %u pairs\n", gid, nFound);
	});

	KERNEL11_CLASSP( pairStore, cl_mem, cl_mem, uint,
			cl_mem, uint, uint,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem,
			oclDEFINES, //"#define PRINTF(a) printf a",
				__kernel void pairStore(
						//configuration
						__global const uint * layer1, __global const uint * layer2, const uint nLayers,
						__global const uint * grid, const uint nSectorsZ, const uint nSectorsPhi,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * oracleOffset, __global const uint * prefixSum,
						// output of pairs
						__global uint2 * pairs, __global uint * pairOffsets)
		{
		size_t thread = get_global_id(0); // thread
		size_t layerPair = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerPairs = get_global_size(1); //total number of processed layer pairings

		uint layer = layer2[layerPair]-1; //outer layer
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint layer2Offset = grid[offset]; //offset of outer layer
		uint i = layer2Offset + thread;
		uint end = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1]; //last hit of outer layer in hit array
		uint nHits2 = end - layer2Offset; //hits in second  layer

		layer = layer1[layerPair]-1; //inner layer
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint nHits1 = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1] - grid[offset]; //number of hits in first layer
		offset = grid[offset]; //beginning of inner layer

		uint pos = prefixSum[event*nLayerPairs*threads + layerPair*threads + thread]; //first position to write
		uint nextThread = prefixSum[event*nLayerPairs*threads + layerPair*threads + thread+1]; //first position of next thread

		//configure oracle
		uint byte = oracleOffset[event*nLayerPairs+layerPair]; //offset in oracle array
		//uint bit = (byte + i*nHits2) % 32;
		//byte += (i*nHits2); byte /= 32;
		//uint sOracle = oracle[byte];

		PRINTF(("%lu-%lu-%lu: from hit2 %u to %u with hits1 %u using memory %u to %u\n", event, layerPair, thread, i, end, nHits1, pos, nextThread));
		for(; i < end; i += threads){
			for(uint j = 0; j < nHits1 && pos < nextThread; ++j){ // pos < prefixSum[id+1] can lead to thread divergence
				//is this a valid triplet?
				uint index = j * nHits2 + i-layer2Offset;
				bool valid = oracle[(byte + index) / 32] & (1 << (index % 32));

				PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u and %u -> %u written at %u\n" : "", event, layerPair, thread, j, i,  index, pos));
					//PRINTF("%lu-%lu-%lu: %u : %u - %u\n", event, layerPair, thread, i-layer1Offset, offset+j,  index);
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
					pairs[pos].x = j + offset;
					pairs[pos].y = i;
				}

				//if(valid)
				//PRINTF("[ %lu ] Written at %i: %i-%i\n", gid, pos, pairs[pos].x,pairs[pos].y);

				//advance pos if valid
				pos += valid;
			}
		}

		if(thread == threads-1){ //store pos in pairOffset array
			pairOffsets[event * nLayerPairs + layerPair + 1] = nextThread;
		}
		});

};
