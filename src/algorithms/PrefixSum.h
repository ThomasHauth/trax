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
	#define MEMORY_BANK_COUNT       (16)  // Adjust to your architecture
	#define LOG2_MEMORY_BANK_COUNT   (4)  // Set to log2(MEMORY_BANK_COUNT)
	#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)

	static const std::string oclDEFINES;

public:

	PrefixSum(clever::context & ctext) :
		ctx(ctext),
		prefixSum(ctext),
		uniformAdd(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "PrefixSum Kernel WorkGroupSize: " << prefixSum.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "PREFIX_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "";}

	cl_event run(cl_mem input, const uint size, uint nThreads);

private:

	std::vector<clever::vector<uint, 1> *> partialSums;

	void createPartialSums(uint size, uint wg);

	cl_event recursiveScan(cl_mem input, uint size, uint wg, uint level);

	KERNEL4_CLASSP( prefixSum, cl_mem, uint, cl_mem, local_param, oclDEFINES,

			__kernel void prefixSum(
					//input
					__global uint * input, const uint size,
					//partial sums
					__global uint * partial,
					// intermeditate data
					__local uint * data )
	{
		uint gid = get_group_id(0);
		uint lid = get_local_id(0);
		uint ls = get_local_size(0);

		uint gOffset = gid*(ls<<1);

		//load elements into buffer
		data[lid + MEMORY_BANK_OFFSET(lid)] = (gOffset + lid) < size ? input[gOffset + lid] : 0;
		data[lid+ls + MEMORY_BANK_OFFSET(lid+ls)] = (gOffset + lid + ls) < size ? input[gOffset + lid + ls] : 0;
		barrier(CLK_LOCAL_MEM_FENCE);

		uint offset = 1;

		// Build the sum in place up the tree.
		for (uint d = ls; d > 0; d >>= 1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lid < d)
			{
				uint ai = offset*((lid<<1)+1)-1;
				uint bi = offset*((lid<<1)+2)-1;

				ai += MEMORY_BANK_OFFSET(ai);
				bi += MEMORY_BANK_OFFSET(bi);

				data[bi] += data[ai];
			}

			offset <<= 1;
		}

		// Scan back down the tree.

		// Clear the last element
		if (lid == 0)
		{

			uint idx = (ls<<1) - 1;
			idx += MEMORY_BANK_OFFSET(idx);
			if(get_num_groups(0) > 1) //no divergence, as all work-items get same branch
				partial[gid] = data[idx];
			data[idx] = 0;
		}

		// Traverse down the tree building the scan in place.
		for (uint d = 1; d <= ls; d <<= 1)
		{
			offset >>= 1;
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lid < d)
			{
				uint ai = offset*((lid<<1)+1)-1;
				uint bi = offset*((lid<<1)+2)-1;

				ai += MEMORY_BANK_OFFSET(ai);
				bi += MEMORY_BANK_OFFSET(bi);

				uint t = data[ai];
				data[ai] = data[bi];
				data[bi] += t;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//store elements in global mem
		if(gOffset + lid < size)
			input[gOffset + lid] = data[lid + MEMORY_BANK_OFFSET(lid)];
		if(gOffset + lid + ls < size) //only in last work-group relevant
			input[gOffset + lid + ls] = data[lid+ls + MEMORY_BANK_OFFSET(lid+ls)];
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

	KERNEL3_CLASS( uniformAdd, cl_mem, uint, cl_mem,

			__kernel void uniformAdd(
					//input
					__global uint * input, const uint size,
					//partial sums
					__global uint * partial )
	{
		uint gid = get_group_id(0);
		uint lid = get_local_id(0);
		uint ls = get_local_size(0);

		uint gOffset = gid*(ls<<1);

		//load addend
		uint add = partial[gid];

		barrier(CLK_LOCAL_MEM_FENCE);
		//add it to input
		if(gOffset + lid < size)
			input[gOffset + lid] += add;
		if(gOffset + lid + ls < size) //only in last work-group relevant
			input[gOffset + lid + ls] += add;
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);

};
