#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/LayerSupplement.h>


using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class HitSorter: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	HitSorter(clever::context & ctext) :
		ctx(ctext),
		sortingKernel(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "Sorting Kernel WorkGroupSize: " << sortingKernel.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "SORT_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "";}

	void run(HitCollectionTransfer & hits, int nThreads, uint maxLayer, const LayerSupplement & layerSupplement)
	{

		clever::vector<uint, 1> layerHits(layerSupplement.getLayerHits(), ctx);
		clever::vector<uint, 1> layerOffsets(layerSupplement.getLayerOffsets(), ctx);

		for(uint layer = 1; layer <= maxLayer; ++layer){
			cl_event evt = sortingKernel.run(
					//configuration
					layer, layerHits.get_mem(), layerOffsets.get_mem(),
					// input
					hits.buffer(GlobalX()), hits.buffer(GlobalY()), hits.buffer(GlobalZ()),
					//aux
					local_param(sizeof(cl_float4), layerSupplement[layer-1].nHits),
					//threads
					nThreads);
		}
	}

	KERNEL7_CLASS( sortingKernel, uint, cl_mem, cl_mem,  cl_mem, cl_mem, cl_mem, local_param,

	uint pow2i(uint exp){
		return 1 << exp;
	}

	uint log2i(uint n){

		uint k = 0;
		while(pow2i(k++) < n);

		return k-1;
	}

	inline float calcSearchCriterion(float4 a){
		//return atan2( sign(a.y) * sqrt(a.x*a.x + a.y*a.y), a.z);
		return a.z;
	}

	__kernel void sortingKernel(
			//configuration
			uint layer, __global const uint * layerHits, __global const uint * layerOffsets,
			// hit input
			__global float * hitGlobalX, __global float * hitGlobalY, __global float * hitGlobalZ,
			// intermeditate data
			__local float4 * aux )
	{
		uint gid = get_global_id(0); // thread
		uint threads = get_global_size(0);

		uint hits = layerHits[layer-1];
		uint offset = 0;//layerOffsets[layer-1];

		//load elements into buffer
		for(uint i = gid; i < hits; i+= threads){
				aux[i] = (float4) (hitGlobalX[offset + i], hitGlobalY[offset + i], hitGlobalZ[offset + i], 0);
				//calculate theta and store it in w
				aux[i].w = calcSearchCriterion(aux[i]); //float3 == float4 -> .w is free
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//sort
		uint paddedSize = 1 << log2i(hits);
		while(paddedSize < hits) paddedSize <<= 1;

		for (uint p = paddedSize >> 1; p > 0; p >>= 1){
			uint r = 0;
			uint d = p;

			for (uint q = paddedSize >> 1; q >= p; q >>= 1){
				for (uint k = gid; k < hits; k += threads){
					if (((k & p) == r) && ((k + d) < hits)){
						if (aux[k].w > aux[k+d].w){
							float4 temp = aux[k];
							aux[k] = aux[k+d];
							aux[k+d] = temp;

						}
					}
				}

				d = q - p;
				r = p;

				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//write back
		for(uint i = gid; i < hits; i+= threads){
			hitGlobalX[offset + i] = aux[i].x;
			hitGlobalY[offset + i] = aux[i].y;
			hitGlobalZ[offset + i] = aux[i].z;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

};
