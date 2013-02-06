#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>


using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class PrefixSum: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	PrefixSum(clever::context & ctext) :
		ctx(ctext),
		prefixSumKernel(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "PrefixSum Kernel WorkGroupSize: " << prefixSumKernel.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "PREFIX_COMPUTE";}
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

	uint run(clever::vector<uint,1> & input, const uint nThreads, const bool storeTotalInLastElement = false){
		return run(input, input.get_count(), nThreads, storeTotalInLastElement);
	}

	uint run(clever::vector<uint,1> & input, const uint size, const uint nThreads, const bool storeTotalInLastElement = false)
	{

		uint lSize = size - storeTotalInLastElement;

		cl_bool clStoreTotalInLastElement = storeTotalInLastElement ? CL_TRUE : CL_FALSE;

		cl_event evt = prefixSumKernel.run(
				input.get_mem(), lSize, clStoreTotalInLastElement, local_param(sizeof(cl_uint), getNextPowerOfTwo(lSize)),
				//threads
				nThreads);

		uint out;
		transfer::downloadScalar(input, out, ctx,true,size-1,1,&evt);

		return out;
	}

	KERNEL4_CLASS( prefixSumKernel, cl_mem, uint, cl_bool, local_param,

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
			__global uint * input, const uint size, uint storeTotalInLastElement,
			// intermeditate data
			__local uint * data )
	{
		uint gid = get_global_id(0); // thread
		uint threads = get_global_size(0);

		uint paddedSize = getNextPowerOfTwo(size);

		//load elements into buffer
		for(uint i = gid; i < size; i+= threads){
			data[i] = input[i];
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

		//store total in size+1 if requested
		if(storeTotalInLastElement && gid == 0){
			input[size] = data[size-1] + input[size-1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//store elements in global mem
		for(uint i = gid; i < size; i+= threads){
			input[i] = data[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

};
