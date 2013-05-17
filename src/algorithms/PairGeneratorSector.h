#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/TripletConfiguration.h>
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
class PairGeneratorSector: public KernelWrapper<PairGeneratorSector>
{

public:

	PairGeneratorSector(clever::context & ctext) :
		KernelWrapper(ctext),
		pairCount(ctext),
		pairStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "FilterKernel WorkGroupSize: " << pairCount.getWorkGroupSize() << std::endl;
		PLOG << "StoreKernel WorkGroupSize: " << pairStore.getWorkGroupSize() << std::endl;
}

	Pairing * run(HitCollection & hits,
				uint nThreads, const TripletConfigurations & layerTriplets, const Grid & grid);

	KERNEL19_CLASSP( pairCount, cl_mem, cl_mem, uint,
			cl_mem,
			cl_float, cl_float, uint,
			cl_float, cl_float, uint,
			cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			cl_mem, cl_mem, cl_mem,
			local_param,
			oclDEFINES,
			__kernel void pairCount(
					//configuration
					__global const uint * layer1, __global const uint * layer2, const uint nLayers,
					__global const uint * grid,
					const float minZ, const float sectorSizeZ , const uint nSectorsZ,
					const float minPhi, const float sectorSizePhi , const uint nSectorsPhi,
					//how many sectors to cover
					__global const uint * pairSpreadZ, __global const uint * pairSpreadPhi,
					//hit data
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global uint * oracle, __global const uint * oracleOffset, __global uint * prefixSum,
					__local uint * lGrid2)
	{
		size_t thread = get_global_id(0); // thread
		size_t layerPair = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer
		size_t nLayerPairs = get_global_size(1); //total number of processed layer pairings

		uint layer = layer1[layerPair]-1; //inner layer
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint layer1Offset = grid[offset]; //offset of inner layer
		uint i = layer1Offset + thread;
		uint end = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1]; //last hit of inner layer in hit array

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u\n", event, layerPair, thread, i, end));

		layer = layer2[layerPair]-1; //outer layer
		uint nHits2 = (nSectorsZ+1)*(nSectorsPhi+1); //temp: number of grid cells
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in grid data structure for outer layer
		//load grid for second layer to local mem
		for(uint i = thread; i < nHits2; i += threads){
			lGrid2[i] = grid[offset + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		nHits2 = lGrid2[nHits2-1] - lGrid2[0]; //number of hits in second layer
		offset = lGrid2[0]; //beginning of outer layer

		PRINTF(("%lu-%lu-%lu: second layer from %u with %u hits\n", event, layerPair, thread, offset, nHits2));

		uint spreadZ = pairSpreadZ[layerPair];
		uint spreadPhi = pairSpreadPhi[layerPair];

		uint oOffset = oracleOffset[event*nLayerPairs+layerPair]; //offset in oracle array
		uint nFound = 0;

		PRINTF(("%lu-%lu-%lu: oracle from %u\n", event, layerPair, thread, oOffset));

		//PRINTF(("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", gid, threads, workload, i, end, (sectorBorders1[2*sector] - sectorBorders1[2*(sector-1)])));

		for(; i < end; i += threads){

			uint zLowSector = floor((hitGlobalZ[i] - minZ) / sectorSizeZ); //register used to store sector
			uint zHighSector = min(zLowSector + spreadZ+1, nSectorsZ); // upper sector border equals sectorNumber + 1
			zLowSector = max(0u, zLowSector-spreadZ); //lower sector border equals sectorNumber

			PRINTF(("%lu-%lu-%lu: hit1 %u:  z = %f -> [%u,%u]\n", event, layerPair, thread, i, hitGlobalZ[i], zLowSector, zHighSector));

			float phi = atan2(hitGlobalY[i], hitGlobalX[i]);

			int phiLowSector= floor((phi - minPhi) / sectorSizePhi) - spreadPhi; // lower phi sector, can underflow
			PRINTF(("phi %f sector %f-%u\n", phi, (phi - minPhi) / sectorSizePhi, phiLowSector));
			uint phiHighSector = phiLowSector + 2*spreadPhi + 1; //higher phi sector, can overflow
			bool wrapAround = phiLowSector < 0 || phiHighSector > (nSectorsPhi + 1); // does wrap around occur?
			PRINTF(("%lu-%lu-%lu: hit1 %u:  phi = %f -> [%i,%u] %s\n", event, layerPair, thread, i, phi, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));
			phiLowSector += (phiLowSector < 0) * (nSectorsPhi); //correct wraparound
			phiHighSector -= (phiHighSector > (nSectorsPhi + 1)) * (nSectorsPhi);

			PRINTF(("%lu-%lu-%lu: hit1 %u:  phi = %f -> [%i,%u] %s\n", event, layerPair, thread, i, phi, phiLowSector, phiHighSector, wrapAround ? " wrapAround" : ""));

			for(uint zSector = zLowSector; zSector < zHighSector; ++ zSector){

				uint zSectorStart = lGrid2[(zSector)*(nSectorsPhi+1)];
				uint zSectorEnd = lGrid2[(zSector+1)*(nSectorsPhi+1)-1];
				uint zSectorLength = zSectorEnd - zSectorStart;

				uint j = lGrid2[zSector*(nSectorsPhi+1)+phiLowSector];
				uint end2 = wrapAround * zSectorEnd + //add end of layer
						lGrid2[zSector*(nSectorsPhi+1)+phiHighSector] //actual end of sector
						                 - wrapAround * (zSectorStart); //substract start of zSector

				PRINTF(("%lu-%lu-%lu: hit2 from %u to %u\n", event, layerPair, thread, j, end2));

				for(; j < end2; ++j){

					++nFound;

					//update oracle
					//           skip to appropriate inner hit
					//								treat phi overflow
					//																beginning of second layer
					PRINTF(("%lu-%lu-%lu: setting bit for %u and %u (%u) -> %u\n", event, layerPair, thread, i-layer1Offset, j - (j >= zSectorEnd) * zSectorLength, j,  (i-layer1Offset)*nHits2 + j - (j >= zSectorEnd) * zSectorLength - offset));
					//PRINTF(("%lu-%lu-%lu: %u : %u - %u\n", event, layerPair, thread, i-layer1Offset, j - (j >= zSectorEnd) * zSectorLength,  (i-layer1Offset)*nHits2 + j - (j >= zSectorEnd) * zSectorLength - offset));
					uint index = (i-layer1Offset)*nHits2 + j - (j >= zSectorEnd) * zSectorLength - offset;
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
			cl_mem, cl_mem, oclDEFINES,
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

		uint layer = layer1[layerPair]-1; //inner layer
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint layer1Offset = grid[offset]; //offset of inner layer
		uint i = layer1Offset + thread;
		uint end = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1]; //last hit of inner layer in hit array

		layer = layer2[layerPair]-1; //inner layer
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1); //offset in hit array
		uint nHits2 = grid[offset + (nSectorsZ+1)*(nSectorsPhi+1)-1] - grid[offset]; //number of hits in second layer
		offset = grid[offset]; //beginning of outer layer

		uint pos = prefixSum[event*nLayerPairs*threads + layerPair*threads + thread]; //first position to write
		uint nextThread = prefixSum[event*nLayerPairs*threads + layerPair*threads + thread+1]; //first position of next thread

		//configure oracle
		uint byte = oracleOffset[event*nLayerPairs+layerPair]; //offset in oracle array
		//uint bit = (byte + i*nHits2) % 32;
		//byte += (i*nHits2); byte /= 32;
		//uint sOracle = oracle[byte];

		PRINTF(("%lu-%lu-%lu: from hit1 %u to %u with hits2 %u using memory %u to %u\n", event, layerPair, thread, i, end, nHits2, pos, nextThread));
		for(; i < end; i += threads){
			for(uint j = 0; j < nHits2 && pos < nextThread; ++j){ // pos < prefixSum[id+1] can lead to thread divergence
				//is this a valid triplet?
				uint index = (i-layer1Offset) * nHits2 + j;
				bool valid = oracle[(byte + index) / 32] & (1 << (index % 32));

				PRINTF((valid ? "%lu-%lu-%lu: valid bit for %u and %u -> %u written at %u\n" : "", event, layerPair, thread, i-layer1Offset, offset+j,  index, pos));
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
					pairs[pos].x = i;
					pairs[pos].y = offset + j;
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
