#pragma once

#include <vector>

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>


using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class BoundarySelection: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	BoundarySelection(clever::context & ctext) :
		ctx(ctext),
		selectionKernel(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "Boundary Selection Kernel WorkGroupSize: " << selectionKernel.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "SELECTION_COMPUTE";}
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

	void run(HitCollectionTransfer & hits, int nThreads, uint maxLayer, const LayerSupplement & layerSupplement, Grid & grid)
	{

		std::vector<float> hBoundaryValues;
		float step = (Grid::MAX_Z - Grid::MIN_Z) / grid.nSectorsZ;
		for(uint i = 0; i <= grid.nSectorsZ; ++i){
			hBoundaryValues.push_back(Grid::MIN_Z + i*step);
		}
		clever::vector<cl_float, 1> boundaryValues(hBoundaryValues, ctx);

		for(uint layer = 1; layer <= maxLayer; ++layer){
			cl_event evt = selectionKernel.run(
					//configuration
					layer, layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()),
					boundaryValues.get_mem(),
					grid.nSectorsZ, grid.nSectorsPhi,
					// input
					hits.buffer(GlobalZ()),
					//aux
					local_param(sizeof(cl_uint), layerSupplement[layer-1].getNHits()), local_param(sizeof(cl_ushort), getNextPowerOfTwo(layerSupplement[layer-1].getNHits()*(grid.nSectorsZ+1))),
					local_param(sizeof(cl_ushort), getNextPowerOfTwo(grid.nSectorsZ+1)),
					//threads
					nThreads);
		}

		/*uint layer, __global const uint * layerHits, __global const uint * layerOffsets,
					__global uint * sectorBoundaries, __global const uint * boundaryValues, uint nSectorsZ, uint nSectorsPhi,
					// hit input
					__global float * hitGlobalZ,
					//local
					__local uint * sBoundaryValues, __local ushort * sGlobalPrefix, __local ushort * sBoundaryPrefix)*/
	}

	KERNEL11_CLASS( selectionKernel, uint, cl_mem, cl_mem,  cl_mem, cl_mem, uint, uint,  cl_mem,local_param,local_param,local_param,

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

	void prefixSum(__local ushort * data, uint size){

		uint gid = get_global_id(0);
		uint threads = get_global_size(0);

		uint paddedSize = getNextPowerOfTwo(size); //data must be of this size

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

	}

	__kernel void selectionKernel(
			//configuration
			uint layer, __global const uint * layerHits, __global const uint * layerOffsets,
			__global uint * sectorBoundaries, __global const uint * boundaryValues, uint nSectorsZ, uint nSectorsPhi,
			// hit input
			__global float * hitGlobalZ,
			//local
			__local uint * sBoundaryValues, __local ushort * sGlobalPrefix, __local ushort * sBoundaryPrefix)
	{
		uint gid = get_global_id(0); // thread
		uint threads = get_global_size(0);

		uint hits = layerHits[layer-1];
		uint offset = layerOffsets[layer-1];

		//load boundary values into buffer
		for(uint i = gid; i <= nSectorsZ; i+= threads){
			sBoundaryValues[i] = boundaryValues[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for(uint i = gid; i < hits; i+= threads){
			float hitZ = hitGlobalZ[offset + i];

			for(uint b = 0; b <= nSectorsZ; ++b){
				sGlobalPrefix[b*hits + i] = hitZ < sBoundaryValues[b];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//compute prefix sum of sGlobalPrefix ==> must be of length 2^x
		prefixSum(sGlobalPrefix, hits*(nSectorsZ+1));

		//store last item of each boundary prefix sum in sBoundaryPrefix
		for(uint i = gid; i <= nSectorsZ; i+= threads){
			sBoundaryPrefix[gid] = sGlobalPrefix[(gid*hits)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		//calculate prefix sum ==> must be length of 2^x
		prefixSum(sGlobalPrefix, nSectorsZ+1);

		//store boundary borders
		for(uint i = gid; i <= nSectorsZ; i+= threads){
			sectorBoundaries[i*(nSectorsPhi+1)] = sGlobalPrefix[i*hits] - sBoundaryPrefix[i];
		}

	}
	);

};
