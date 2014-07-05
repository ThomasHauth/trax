#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/EventSupplement.h>
#include <datastructures/LayerSupplement.h>
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
class GridBuilder: public KernelWrapper<GridBuilder>
{

public:

	GridBuilder(clever::context & ctext) :
		KernelWrapper(ctext),
		gridCount(ctext),
		gridNoLocalCount(ctext),
		gridStore(ctext),
		gridWrittenLocalStore(ctext),
		gridNoLocalStore(ctext)
{
		// create the buffers this algorithm will need to run
		PLOG << "Grid building Kernel WorkGroupSize: " << gridCount.getWorkGroupSize() << std::endl;
}

	cl_ulong run(HitCollection & hits, uint nThreads, const EventSupplement & eventSupplement, const LayerSupplement & layerSupplement, Grid & grid);

	void printGrid(const Grid & grid);

	void verifyGrid(HitCollection & hits, const Grid & grid);

	KERNEL_CLASS( gridCount,

	__kernel void gridCount(
			//configuration
			__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint nLayers,
			//grid data
			__global uint * grid,
			//grid configuration
			float minZ, float sectorSizeZ, uint nSectorsZ, float minPhi, float sectorSizePhi, uint nSectorsPhi,
			// hit input
			__global float * hitGlobalX, __global float * hitGlobalY, __global float * hitGlobalZ,
			//local grid buffer
			__local uint * lGrid)
	{
		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer

		uint hits = (nSectorsZ+1)*(nSectorsPhi+1);
		//initialize local buffer to zero
		for(uint i = thread; i < hits; i += threads){
			lGrid[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		hits = layerHits[event*nLayers+layer];
		uint offset = eventOffsets[event];
		offset += layerOffsets[event*nLayers+layer];

		for(uint i = thread; i < hits; i += threads){
			uint zSector = floor((hitGlobalZ[offset + i] - minZ) / sectorSizeZ);
			float phi = atan2(hitGlobalY[offset + i] , hitGlobalX[offset + i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			atomic_inc(&lGrid[zSector*(nSectorsPhi+1) + phiSector]);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		hits = (nSectorsZ+1)*(nSectorsPhi+1);
		offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1);
		//copy back local grid to global one
		for(uint i = thread; i < hits; i += threads){
			atomic_add(&grid[offset + i], lGrid[i]);
		}

		/*barrier(CLK_LOCAL_MEM_FENCE);
		if(thread == 0){
			printf("Event %lu - Layer %lu\n", event, layer);

			printf("\t");
			for(uint z = 0; z < (nSectorsZ+1); ++z){
				printf("%.0f\t", minZ + z * sectorSizeZ);
			}
			printf("\n");

			for(uint p = 0; p < (nSectorsPhi+1); ++p){
				printf("%.3f\t", minPhi + p * sectorSizePhi);
				for(uint z = 0; z < (nSectorsZ+1); ++z){
					printf("%u\t", sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+z*(nSectorsPhi+1) + p]);
				}
				printf("\n");
			}
		}*/

	}, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint, cl_mem, cl_mem, cl_mem, local_param);

	KERNEL_CLASS( gridNoLocalCount,

			__kernel void gridNoLocalCount(
					//configuration
					__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint nLayers,
					//grid data
					__global uint * grid,
					//grid configuration
					float minZ, float sectorSizeZ, uint nSectorsZ, float minPhi, float sectorSizePhi, uint nSectorsPhi,
					// hit input
					__global float * hitGlobalX, __global float * hitGlobalY, __global float * hitGlobalZ)
	{
		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_local_size(0); //threads per layer

		uint hits = (nSectorsZ+1)*(nSectorsPhi+1);
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1);
		__global uint * lGrid = &grid[offset];

		hits = layerHits[event*nLayers+layer];
		offset = eventOffsets[event];
		offset += layerOffsets[event*nLayers+layer];

		for(uint i = thread; i < hits; i += threads){
			uint zSector = floor((hitGlobalZ[offset + i] - minZ) / sectorSizeZ);
			float phi = atan2(hitGlobalY[offset + i] , hitGlobalX[offset + i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			atomic_inc(&lGrid[zSector*(nSectorsPhi+1) + phiSector]);
		}

		/*barrier(CLK_LOCAL_MEM_FENCE);
			if(thread == 0){
				printf("Event %lu - Layer %lu\n", event, layer);

				printf("\t");
				for(uint z = 0; z < (nSectorsZ+1); ++z){
					printf("%.0f\t", minZ + z * sectorSizeZ);
				}
				printf("\n");

				for(uint p = 0; p < (nSectorsPhi+1); ++p){
					printf("%.3f\t", minPhi + p * sectorSizePhi);
					for(uint z = 0; z < (nSectorsZ+1); ++z){
						printf("%u\t", sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+z*(nSectorsPhi+1) + p]);
					}
					printf("\n");
				}
			}*/

	}, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint, cl_mem, cl_mem, cl_mem);

	KERNEL_CLASS( gridStore,

	__kernel void gridStore(
			//configuration
			__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint nLayers,
			//grid data
			__global const uint * grid,
			//grid configuration
			const float minZ, const float sectorSizeZ, const uint nSectorsZ, const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
			// hit input
			__global const float * ihitGlobalX, __global const float * ihitGlobalY, __global const float * ihitGlobalZ,
			__global const uint * iDetectorLayer, __global const uint * iDetectorId, __global const uint * iHitId, __global const uint * iEventNumber,
			//hit output
			__global float * ohitGlobalX, __global float * ohitGlobalY, __global float * ohitGlobalZ,
			__global uint * oDetectorLayer, __global uint * oDetectorId, __global uint * oHitId, __global uint * oEventNumber,
			// intermeditate data
			__local uint * lGrid, __local uint * written )
	{

		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_global_size(0); //threads per layer

		uint hits = (nSectorsZ+1)*(nSectorsPhi+1);
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1);
		//load grid to local mem
		for(uint i = thread; i < hits; i += threads){
			lGrid[i] = grid[offset + i];
			written[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		offset = eventOffsets[event];
		offset += layerOffsets[event*nLayers+layer];
		hits = offset + layerHits[event*nLayers+layer];

		/*if(thread == 0){
			printf("Event %lu - Layer %lu\n", event, layer);

			printf("\t");
			for(uint z = 0; z < (nSectorsZ+1); ++z){
				printf("%.0f\t", minZ + z * sectorSizeZ);
			}
			printf("\n");

			for(uint p = 0; p < (nSectorsPhi+1); ++p){
				printf("%.3f\t", minPhi + p * sectorSizePhi);
				for(uint z = 0; z < (nSectorsZ+1); ++z){
					printf("%u\t", sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+z*(nSectorsPhi+1) + p]);
				}
				printf("\n");
			}

			for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
				printf("%u : %u  ", i, lGrid[i]);
			}
			printf("\n");
			for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
				printf("%u : %u  ", i, written[i]);
			}
			printf("\n");
		}
		barrier(CLK_LOCAL_MEM_FENCE);*/

		for(uint i = offset + thread; i < hits; i += threads){
			uint zSector = floor((ihitGlobalZ[i] - minZ) / sectorSizeZ);
			float phi = atan2(ihitGlobalY[i] , ihitGlobalX[i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			uint w = lGrid[zSector*(nSectorsPhi+1) + phiSector];
			w += atomic_inc(&written[zSector*(nSectorsPhi+1) + phiSector]);

			ohitGlobalX[w] = ihitGlobalX[i];
			ohitGlobalY[w] = ihitGlobalY[i];
			ohitGlobalZ[w] = ihitGlobalZ[i];
			oDetectorLayer[w] = iDetectorLayer[i];
			oDetectorId[w] = iDetectorId[i];
			oHitId[w] = iHitId[i];
			oEventNumber[w] = iEventNumber[i];
		}

	}, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint,
	   cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, 
           cl_mem, cl_mem, cl_mem, cl_mem, local_param, local_param);

	KERNEL_CLASS( gridWrittenLocalStore,

			__kernel void gridWrittenLocalStore(
					//configuration
					__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint nLayers,
					//grid data
					__global const uint * grid,
					//grid configuration
					const float minZ, const float sectorSizeZ, const uint nSectorsZ, const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
					// hit input
					__global const float * ihitGlobalX, __global const float * ihitGlobalY, __global const float * ihitGlobalZ,
					__global const uint * iDetectorLayer, __global const uint * iDetectorId, __global const uint * iHitId, __global const uint * iEventNumber,
					//hit output
					__global float * ohitGlobalX, __global float * ohitGlobalY, __global float * ohitGlobalZ,
					__global uint * oDetectorLayer, __global uint * oDetectorId, __global uint * oHitId, __global uint * oEventNumber,
					// intermeditate data
					 __local uint * written )
	{

		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_global_size(0); //threads per layer

		uint hits = (nSectorsZ+1)*(nSectorsPhi+1);
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1);
		__global const uint * lGrid = &grid[offset];
		//load grid to local mem
		for(uint i = thread; i < hits; i += threads){
			written[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		offset = eventOffsets[event];
		offset += layerOffsets[event*nLayers+layer];
		hits = offset + layerHits[event*nLayers+layer];

		/*if(thread == 0){
			printf("Event %lu - Layer %lu\n", event, layer);

			printf("\t");
			for(uint z = 0; z < (nSectorsZ+1); ++z){
				printf("%.0f\t", minZ + z * sectorSizeZ);
			}
			printf("\n");

			for(uint p = 0; p < (nSectorsPhi+1); ++p){
				printf("%.3f\t", minPhi + p * sectorSizePhi);
				for(uint z = 0; z < (nSectorsZ+1); ++z){
					printf("%u\t", sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+z*(nSectorsPhi+1) + p]);
				}
				printf("\n");
			}

			for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
				printf("%u : %u  ", i, lGrid[i]);
			}
			printf("\n");
			for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
				printf("%u : %u  ", i, written[i]);
			}
			printf("\n");
		}
		barrier(CLK_LOCAL_MEM_FENCE);*/

		for(uint i = offset + thread; i < hits; i += threads){
			uint zSector = floor((ihitGlobalZ[i] - minZ) / sectorSizeZ);
			float phi = atan2(ihitGlobalY[i] , ihitGlobalX[i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			uint w = lGrid[zSector*(nSectorsPhi+1) + phiSector];
			w += atomic_inc(&written[zSector*(nSectorsPhi+1) + phiSector]);

			ohitGlobalX[w] = ihitGlobalX[i];
			ohitGlobalY[w] = ihitGlobalY[i];
			ohitGlobalZ[w] = ihitGlobalZ[i];
			oDetectorLayer[w] = iDetectorLayer[i];
			oDetectorId[w] = iDetectorId[i];
			oHitId[w] = iHitId[i];
			oEventNumber[w] = iEventNumber[i];
		}

	}, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint,
           cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
           cl_mem, cl_mem, cl_mem, cl_mem, local_param);

	KERNEL_CLASS( gridNoLocalStore,

			__kernel void gridNoLocalStore(
					//configuration
					__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint nLayers,
					//grid data
					__global const uint * grid,
					//grid configuration
					const float minZ, const float sectorSizeZ, const uint nSectorsZ, const float minPhi, const float sectorSizePhi, const uint nSectorsPhi,
					// hit input
					__global const float * ihitGlobalX, __global const float * ihitGlobalY, __global const float * ihitGlobalZ,
					__global const uint * iDetectorLayer, __global const uint * iDetectorId, __global const uint * iHitId, __global const uint * iEventNumber,
					//hit output
					__global float * ohitGlobalX, __global float * ohitGlobalY, __global float * ohitGlobalZ,
					__global uint * oDetectorLayer, __global uint * oDetectorId, __global uint * oHitId, __global uint * oEventNumber,
					// intermeditate data
					__global uint * gWritten )
	{

		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_global_size(0); //threads per layer

		uint hits = (nSectorsZ+1)*(nSectorsPhi+1);
		uint offset = event*nLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1);
		__global const uint * lGrid = &grid[offset];
		__global uint * written = &gWritten[offset];

		offset = eventOffsets[event];
		offset += layerOffsets[event*nLayers+layer];
		hits = offset + layerHits[event*nLayers+layer];

		/*if(thread == 0){
				printf("Event %lu - Layer %lu\n", event, layer);

				printf("\t");
				for(uint z = 0; z < (nSectorsZ+1); ++z){
					printf("%.0f\t", minZ + z * sectorSizeZ);
				}
				printf("\n");

				for(uint p = 0; p < (nSectorsPhi+1); ++p){
					printf("%.3f\t", minPhi + p * sectorSizePhi);
					for(uint z = 0; z < (nSectorsZ+1); ++z){
						printf("%u\t", sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+z*(nSectorsPhi+1) + p]);
					}
					printf("\n");
				}

				for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
					printf("%u : %u  ", i, lGrid[i]);
				}
				printf("\n");
				for(uint i = 0; i < (nSectorsZ+1)*(nSectorsPhi+1); ++i){
					printf("%u : %u  ", i, written[i]);
				}
				printf("\n");
			}
			barrier(CLK_LOCAL_MEM_FENCE);*/

		for(uint i = offset + thread; i < hits; i += threads){
			uint zSector = floor((ihitGlobalZ[i] - minZ) / sectorSizeZ);
			float phi = atan2(ihitGlobalY[i] , ihitGlobalX[i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			uint w = lGrid[zSector*(nSectorsPhi+1) + phiSector];
			w += atomic_inc(&written[zSector*(nSectorsPhi+1) + phiSector]);

			ohitGlobalX[w] = ihitGlobalX[i];
			ohitGlobalY[w] = ihitGlobalY[i];
			ohitGlobalZ[w] = ihitGlobalZ[i];
			oDetectorLayer[w] = iDetectorLayer[i];
			oDetectorId[w] = iDetectorId[i];
			oHitId[w] = iHitId[i];
			oEventNumber[w] = iEventNumber[i];
		}

	}, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint,
	   cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
           cl_mem, cl_mem, cl_mem, cl_mem, cl_mem);
};
