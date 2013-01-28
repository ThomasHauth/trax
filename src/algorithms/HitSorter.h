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
					hits.buffer(DetectorLayer()), hits.buffer(DetectorId()), hits.buffer(HitId()), hits.buffer(EventNumber()),
					//aux
					local_param(sizeof(cl_uint), layerSupplement[layer-1].nHits), local_param(sizeof(cl_float), layerSupplement[layer-1].nHits),
					local_param(sizeof(cl_float3), layerSupplement[layer-1].nHits), local_param(sizeof(cl_uint4), layerSupplement[layer-1].nHits),
					//threads
					nThreads);
		}

		ctx.finish_default_queue();
	}

	KERNEL14_CLASS( sortingKernel, uint, cl_mem, cl_mem,  cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,  cl_mem, cl_mem, local_param,local_param,local_param,local_param,

	uint pow2i(uint exp){
		return 1 << exp;
	}

	uint log2i(uint n){

		uint k = 0;
		while(pow2i(k++) < n);

		return k-1;
	}

	inline float calcSearchCriterion(float3 a){
		//return atan2( sign(a.y) * sqrt(a.x*a.x + a.y*a.y), a.z);
		return a.z;
	}

	__kernel void sortingKernel(
			//configuration
			uint layer, __global const uint * layerHits, __global const uint * layerOffsets,
			// hit input
			__global float * hitGlobalX, __global float * hitGlobalY, __global float * hitGlobalZ,
			__global uint * DetectorLayer, __global uint * DetectorId, __global uint * HitId, __global uint * EventNumber,
			// intermeditate data
			__local uint * sIdx, __local float * sSortCrit, __local float3 * sPos, __local uint4 * sData )
	{
		uint gid = get_global_id(0); // thread
		uint threads = get_global_size(0);

		uint hits = layerHits[layer-1];
		uint offset = layerOffsets[layer-1];

		//load elements into buffer
		for(uint i = gid; i < hits; i+= threads){
				sPos[i] = (float3) (hitGlobalX[offset + i], hitGlobalY[offset + i], hitGlobalZ[offset + i]);
				sData[i] = (uint4) (DetectorLayer[offset + i], DetectorId[offset + i], HitId[offset + i], EventNumber[offset + i]);

				sIdx[i] = i;
				//calculate theta and store it in w
				//sSortCrit[i] = calcSearchCriterion(aux); //float3 == float4 -> .w is free
				sSortCrit[i] = hitGlobalZ[offset + i];
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
						if (sSortCrit[k] > sSortCrit[k+d]){
							float tempCrit = sSortCrit[k];
							sSortCrit[k] = sSortCrit[k+d];
							sSortCrit[k+d] = tempCrit;
							//
							uint tempIdx = sIdx[k];
							sIdx[k] = sIdx[k+d];
							sIdx[k+d] = tempIdx;
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
			hitGlobalX[offset + i] = sPos[sIdx[i]].x;
			hitGlobalY[offset + i] = sPos[sIdx[i]].y;
			hitGlobalZ[offset + i] = sPos[sIdx[i]].z;
			DetectorLayer[offset + i] = sData[sIdx[i]].x;
			DetectorId[offset + i] = sData[sIdx[i]].y;
			HitId[offset + i] = sData[sIdx[i]].z;
			EventNumber[offset + i] = sData[sIdx[i]].w;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

};
