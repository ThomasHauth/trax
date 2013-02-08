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
class HitSorterPhi: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	HitSorterPhi(clever::context & ctext) :
		ctx(ctext),
		sortingPhi_kernel(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "Sorting Kernel WorkGroupSize: " << sortingPhi_kernel.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "SORT_PHI_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "";}

	cl_ulong run(HitCollection & hits, uint nThreads, uint maxLayer, const LayerSupplement & layerSupplement, const Grid & grid)
	{

		nThreads = min(nThreads, sortingPhi_kernel.getWorkGroupSize());

		std::vector<cl_event> events;

		for(uint layer = 1; layer <= maxLayer; ++layer){
			cl_event evt = sortingPhi_kernel.run(
					//configuration
					layer, layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()),
					GridConfig::MIN_Z, grid.config.sectorSizeZ,
					// input
					hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
					hits.transfer.buffer(DetectorLayer()), hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()), hits.transfer.buffer(EventNumber()),
					//aux
					local_param(sizeof(cl_uint), layerSupplement[layer-1].getNHits()), local_param(sizeof(cl_float), layerSupplement[layer-1].getNHits()),
					local_param(sizeof(cl_float3), layerSupplement[layer-1].getNHits()), local_param(sizeof(cl_uint4), layerSupplement[layer-1].getNHits()),
					//threads
					nThreads);
			events.push_back(evt);
		}

		hits.transfer.fromDevice(ctx, hits, &events);

		cl_ulong runtime = 0;
		for(cl_event evt : events){
			profile_info pinfo = ctx.report_profile(evt);
			runtime += pinfo.runtime();
		}

		return runtime;
	}

	KERNEL16_CLASS( sortingPhi_kernel, uint, cl_mem, cl_mem, cl_float, cl_float, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,  cl_mem, cl_mem, local_param,local_param,local_param,local_param,

	uint pow2i(uint exp){
		return 1 << exp;
	}

	uint log2i(uint n){

		uint k = 0;
		while(pow2i(k++) < n);

		return k-1;
	}

	__kernel void sortingPhi_kernel(
			//configuration
			uint layer, __global const uint * layerHits, __global const uint * layerOffsets,
			float minZ, float sectorSizeZ,
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
				//calculate phi -- add 2PI for each Z-Sector to perform one pass sorting
				sSortCrit[i] = atan2(sPos[i].y, sPos[i].x) + floor((sPos[i].z - minZ) / sectorSizeZ)*2*M_PI_F;
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
