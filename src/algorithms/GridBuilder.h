#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/EventSupplement.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>


using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class GridBuilder: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	GridBuilder(clever::context & ctext) :
		ctx(ctext),
		gridBuilding_kernel(ctext),
		gridStore_kernel(ctext),
		prefixSumKernel(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "Grid building Kernel WorkGroupSize: " << gridBuilding_kernel.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "GRID_BUILD_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "";}

	uint getNextPowerOfTwo(uint n){
		n--;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
		n++;

		return n;
	}

	cl_ulong run(HitCollection & hits, uint nThreads, const EventSupplement & eventSupplement, const LayerSupplement & layerSupplement, Grid & grid)
	{

		//store buffers to input data
		cl_mem x = hits.transfer.buffer(GlobalX()); cl_mem y =  hits.transfer.buffer(GlobalY()); cl_mem z =  hits.transfer.buffer(GlobalZ());
		cl_mem layer= hits.transfer.buffer(DetectorLayer()); cl_mem detId = hits.transfer.buffer(DetectorId()); cl_mem hitId = hits.transfer.buffer(HitId()); cl_mem evtId = hits.transfer.buffer(EventNumber());

		std::vector<cl_event> events;

		std::cout << "Grid count kernel" << std::endl;
		cl_event evt = gridBuilding_kernel.run(
				//configuration
				eventSupplement.transfer.buffer(Offset()), layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()), layerSupplement.nLayers,
				//grid data
				grid.transfer.buffer(Boundary()),
				//grid configuration
				grid.config.MIN_Z, grid.config.sectorSizeZ, grid.config.nSectorsZ,
				grid.config.MIN_PHI, grid.config.sectorSizePhi,grid.config.nSectorsPhi,
				// hit input
				x, y, z,
				//work item config
				range(nThreads,layerSupplement.nLayers, eventSupplement.nEvents),
				range(nThreads,1, 1));
		events.push_back(evt);

		//run prefix sum on grid boundaries
		std::cout << "Grid prefix kernel" << std::endl;
		uint gridSize = (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1);
		evt = prefixSumKernel.run(grid.transfer.buffer(Boundary()), gridSize, local_param(sizeof(cl_uint), getNextPowerOfTwo(gridSize)),
				range(nThreads,layerSupplement.nLayers, eventSupplement.nEvents), range(nThreads,1, 1));
		events.push_back(evt);
		grid.transfer.fromDevice(ctx,grid, &events);

		//output grid
		for(uint e = 0; e < eventSupplement.nEvents; ++e){
			std::cout << "Event: " << e << std::endl;
			for(uint l = 1; l <= layerSupplement.nLayers; ++l){
				std::cout << "Layer: " << l << std::endl;

				//output z boundaries
				std::cout << "z/phi\t\t";
				for(uint i = 0; i <= grid.config.nSectorsZ; ++i){
					std::cout << grid.config.boundaryValuesZ[i] << "\t";
				}
				std::cout << std::endl;

				LayerGrid layerGrid(grid, l, layerSupplement.nLayers,e);
				for(uint p = 0; p <= grid.config.nSectorsPhi; ++p){
					std::cout << std::setprecision(3) << grid.config.boundaryValuesPhi[p] << "\t\t";
					for(uint z = 0; z <= grid.config.nSectorsZ; ++z){
						std::cout << layerGrid(z,p) << "\t";
					}
					std::cout << std::endl;
				}
			}
		}

		clever::vector<uint, 1> aux_written(0, grid.size(), ctx);

		//allocate new buffers for ouput
		hits.transfer.initBuffers(ctx, hits);

		std::cout << "Grid store kernel" << std::endl;
		evt = gridStore_kernel.run(
				//configuration
				eventSupplement.transfer.buffer(Offset()), layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()), layerSupplement.nLayers,
				//grid data
				grid.transfer.buffer(Boundary()),
				//grid configuration
				grid.config.MIN_Z, grid.config.sectorSizeZ, grid.config.nSectorsZ,
				grid.config.MIN_PHI, grid.config.sectorSizePhi,grid.config.nSectorsPhi,
				// hit input
				x,y,z,
				layer,detId, hitId, evtId,
				// hit output
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
				hits.transfer.buffer(DetectorLayer()), hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()), hits.transfer.buffer(EventNumber()),
				//aux
				aux_written.get_mem(),
				//work item config
				range(nThreads,layerSupplement.nLayers, eventSupplement.nEvents));

		//download updated hit data
		hits.transfer.fromDevice(ctx,hits);

		//delete old buffers
		ctx.release_buffer(x); ctx.release_buffer(y); ctx.release_buffer(z);
		ctx.release_buffer(layer); ctx.release_buffer(detId); ctx.release_buffer(hitId); ctx.release_buffer(evtId);

		return 0;
	}

	KERNEL14_CLASS( gridBuilding_kernel, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint, cl_mem, cl_mem, cl_mem,

	__kernel void gridBuilding_kernel(
			//configuration
			__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint maxLayers,
			//grid data
			__global uint * sectorBoundaries,
			//grid configuration
			float minZ, float sectorSizeZ, uint nSectorsZ, float minPhi, float sectorSizePhi, uint nSectorsPhi,
			// hit input
			__global float * hitGlobalX, __global float * hitGlobalY, __global float * hitGlobalZ )
	{
		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_global_size(0); //threads per layer
		uint hits = layerHits[event*maxLayers+layer];

		uint workload = hits / threads + 1;
		uint i = thread * workload;
		uint end = min(i + workload, hits); // for last thread, if not a full workload is present

		uint tOffset = eventOffsets[event];
		tOffset += layerOffsets[event*maxLayers+layer];

		for(; i < end; ++i){
			uint zSector = floor((hitGlobalZ[tOffset + i] - minZ) / sectorSizeZ);
			float phi = atan2(hitGlobalY[tOffset + i] , hitGlobalX[tOffset + i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			atomic_inc(&sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+zSector*(nSectorsPhi+1) + phiSector]);
		}


		barrier(CLK_LOCAL_MEM_FENCE);
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
		}

	}
	);

	KERNEL26_CLASS( gridStore_kernel, cl_mem, cl_mem, cl_mem, uint,  cl_mem, float, float, uint, float, float,  uint,
			cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,

	__kernel void gridStore_kernel(
			//configuration
			__global const uint * eventOffsets, __global const uint * layerHits, __global const uint * layerOffsets, uint maxLayers,
			//grid data
			__global uint * sectorBoundaries,
			//grid configuration
			float minZ, float sectorSizeZ, uint nSectorsZ, float minPhi, float sectorSizePhi, uint nSectorsPhi,
			// hit input
			__global float * ihitGlobalX, __global float * ihitGlobalY, __global float * ihitGlobalZ,
			__global uint * iDetectorLayer, __global uint * iDetectorId, __global uint * iHitId, __global uint * iEventNumber,
			//hit output
			__global float * ohitGlobalX, __global float * ohitGlobalY, __global float * ohitGlobalZ,
			__global uint * oDetectorLayer, __global uint * oDetectorId, __global uint * oHitId, __global uint * oEventNumber,
			// intermeditate data
			__global uint * written )
	{

		size_t thread = get_global_id(0); // thread
		size_t layer = get_global_id(1); //layer
		size_t event = get_global_id(2); //event

		size_t threads = get_global_size(0); //threads per layer
		uint hits = layerHits[event*maxLayers+layer];

		uint workload = hits / threads + 1;
		uint i = thread * workload;
		uint end = min(i + workload, hits); // for last thread, if not a full workload is present

		uint tOffset = eventOffsets[event];
		tOffset += layerOffsets[event*maxLayers+layer];

		barrier(CLK_LOCAL_MEM_FENCE);
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
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for(; i < end; ++i){
			uint zSector = floor((ihitGlobalZ[tOffset + i] - minZ) / sectorSizeZ);
			float phi = atan2(ihitGlobalY[tOffset + i] , ihitGlobalX[tOffset + i]);
			uint phiSector= floor((phi - minPhi) / sectorSizePhi);

			uint w = sectorBoundaries[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+zSector*(nSectorsPhi+1) + phiSector];
			w += atomic_inc(&written[event*maxLayers*(nSectorsZ+1)*(nSectorsPhi+1)+layer*(nSectorsZ+1)*(nSectorsPhi+1)+zSector*(nSectorsPhi+1) + phiSector]);

			ohitGlobalX[tOffset + w] = ihitGlobalX[tOffset + i];
			ohitGlobalY[tOffset + w] = ihitGlobalY[tOffset + i];
			ohitGlobalZ[tOffset + w] = ihitGlobalZ[tOffset + i];
			oDetectorLayer[tOffset + w] = iDetectorLayer[tOffset + i];
			oDetectorId[tOffset + w] = iDetectorId[tOffset + i];
			oHitId[tOffset + w] = iHitId[tOffset + i];
			oEventNumber[tOffset + w] = iEventNumber[tOffset + i];
		}

	}
	);

	KERNEL3_CLASS( prefixSumKernel, cl_mem, uint, local_param,

	inline uint getNextPowerOfTwo(uint n){
		n--;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
		n++;

		return n;
	}

	__kernel void prefixSumKernel(
			//configuration
			__global uint * input, const uint size,
			// intermeditate data
			__local uint * data )
	{
		size_t gid = get_global_id(0); // thread
		size_t threads = get_global_size(0);

		size_t layer = get_global_id(1); //layer
		size_t nLayers = get_global_size(1); //layers

		size_t event = get_global_id(2); //event

		uint paddedSize = getNextPowerOfTwo(size);

		//load elements into buffer
		for(uint i = gid; i < size; i+= threads){
			data[i] = input[event*nLayers*size + layer*size + i];
		}
		//initialize padding with zero [ OpenCL 1.2 extension cl_khr_initialize_memory allows automatic initialization in the future
		for(uint i = size + gid; i < paddedSize; i += threads){
			data[i] = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		uint offset = 1;

		// Build the sum in place up the tree.
		for (uint d = paddedSize>>1; d > 0; d >>= 1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);

			for (uint p = gid; p < paddedSize; p += threads)
			{

				if (p < d)
				{
					uint ai = offset*((p<<1)+1)-1;
					uint bi = offset*((p<<1)+2)-1;

					data[bi] += data[ai];
				}
			}

			offset <<= 1;
		}

		// Scan back down the tree.

		// Clear the last element
		if (gid == 0)
		{
			data[paddedSize - 1] = 0;
		}

		// Traverse down the tree building the scan in place.
		for (uint d = 1; d < paddedSize; d <<= 1)
		{
			offset >>= 1;
			barrier(CLK_LOCAL_MEM_FENCE);

			for (uint p = gid; p < paddedSize; p += threads)
			{
				if (p < d)
				{
					uint ai = offset*((p<<1)+1)-1;
					uint bi = offset*((p<<1)+2)-1;

					uint t = data[ai];
					data[ai] = data[bi];
					data[bi] += t;
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//store elements in global mem
		for(uint i = gid; i < size; i+= threads){
			input[event*nLayers*size + layer*size + i] = data[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

};
