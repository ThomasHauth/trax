#include <boost/noncopyable.hpp>
#include <clever/clever.hpp>
#include <iostream>
#include <iomanip>

using namespace clever;
using namespace std;

#define MEMORY_BANK_COUNT       (16)  // Adjust to your architecture
#define LOG2_MEMORY_BANK_COUNT   (4)  // Set to log2(MEMORY_BANK_COUNT)
#define ELIMINATE_CONFLICTS      (0)  // Enable for slow address calculation, but zero bank conflicts

////////////////////////////////////////////////////////////////////////////////////////////////////

#if (ELIMINATE_CONFLICTS)
#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT + (index) >> (2*LOG2_MEMORY_BANK_COUNT))
#else
#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

class WGscan{

public:
	WGscan(clever::context & ctext  )
		: krnl ( ctext ){ }


	KERNEL3_CLASS( krnl, cl_mem, uint, local_param,

	__kernel void krnl(
			//input
			__global uint * input, const uint size,
			// intermeditate data
			__local uint * data )
	{
		uint gid = get_group_id(0);
		uint lid = get_local_id(0);
		uint ls = get_local_size(0);

		uint gOffset = gid*(ls<<1);

		//load elements into buffer
		data[lid] = input[gOffset + lid];
		data[lid+ls] = (gOffset + lid + ls) < size ? input[gOffset + lid + ls] : 0; //only in last work-group relevant
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

				data[bi] += data[ai];
			}

			offset <<= 1;
		}

		// Scan back down the tree.

		// Clear the last element
		if (lid == 0)
		{
			data[(ls<<1) - 1] = 0;
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

				uint t = data[ai];
				data[ai] = data[bi];
				data[bi] += t;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//store elements in global mem
		input[gOffset + lid] = data[lid];
		if(gOffset + lid + ls < size) //only in last work-group relevant
			input[gOffset + lid + ls] = data[lid+ls];
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);
};

class WGscanStore{

public:
	WGscanStore(clever::context & ctext  )
		: krnl ( ctext ){ }


	KERNEL4_CLASS( krnl, cl_mem, uint, cl_mem, local_param,

	__kernel void krnl(
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
		data[lid] = input[gOffset + lid];
		data[lid+ls] = (gOffset + lid + ls) < size ? input[gOffset + lid + ls] : 0; //only in last work-group relevant
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

				data[bi] += data[ai];
			}

			offset <<= 1;
		}

		// Scan back down the tree.

		// Clear the last element
		if (lid == 0)
		{
			partial[gid] = data[(ls<<1)-1];
			data[(ls<<1) - 1] = 0;
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

				uint t = data[ai];
				data[ai] = data[bi];
				data[bi] += t;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//store elements in global mem
		input[gOffset + lid] = data[lid];
		if(gOffset + lid + ls < size) //only in last work-group relevant
			input[gOffset + lid + ls] = data[lid+ls];
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);
};

class UniformAdd{

public:
	UniformAdd(clever::context & ctext  )
		: krnl ( ctext ){ }


	KERNEL3_CLASS( krnl, cl_mem, uint, cl_mem,

	__kernel void krnl(
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
		input[gOffset + lid] += add;
		if(gOffset + lid + ls < size) //only in last work-group relevant
			input[gOffset + lid + ls] += add;
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);
};

//////////////////////////////////////////////////
//Global Variables

clever::context *contx;
WGscan * wgScan;
WGscanStore * wgScanStore;
UniformAdd * uniformAdd;

//////////////////////////////////////////////////

//#define DEBUG_OUT

void printVector(std::vector<uint> in){

#ifdef DEBUG_OUT
	for(uint i = 0; i < in.size(); ++i){
		std::cout <<  std::setw(4) << std::setfill(' ') << i << ":"
				<< std::setw(4) << std::setfill(' ') << in[i] << "   ";
		if(i < in.size()-1 && in[i] > in[i+1])
			std::cout << std::endl << std::endl;
	}

	std::cout << std::endl;
#endif

}

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

void recursiveScan(cl_mem input, uint size, uint wg){

	std::cout << "Recursively scanning " << size << " elements with max "
			  << wg << " work-items ";

	// nGroups = size/(2 * workGroupSize)
	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));

	std::cout << "in " << nGroups << " work-groups" << std::endl;

	if(nGroups == 1){
		//only one work-group needed, base case
		uint localSize = std::max(1.0f, ceil(((float) size)/2));
		localSize = getNextPowerOfTwo(localSize);

		std::cout << "Base case: global size = local size: " << localSize << std::endl;

		wgScan->krnl.run(input, size, local_param(sizeof(cl_uint), (wg<<1)), range(localSize), range(localSize) );

		return;
	} else {
		//scan with sums into partial
		clever::vector<uint, 1> partial(0, nGroups, *contx);
		std::cout << "Recursive case: allocating " << nGroups << " partial sums ";
		std::cout << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;

		wgScanStore->krnl.run(input, size, partial.get_mem(), local_param(sizeof(cl_uint), (wg<<1)), range(wg*nGroups), range(wg) );
		contx->finish_default_queue();

		std::vector<uint> lPartial(nGroups);
		transfer::download(partial, lPartial,*contx, true);
		printVector(lPartial);

		recursiveScan(partial.get_mem(), nGroups, wg);

		contx->finish_default_queue();

		transfer::download(partial, lPartial,*contx, true);
		printVector(lPartial);

		std::cout << "Adding " << nGroups << " partial sums to " << size << " inputs "
				  << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;
		uniformAdd->krnl.run(input, size, partial.get_mem(), range(wg*nGroups), range(wg));
	}

}

int main(int argc, char **argv) {

	bool useCPU = true;

	std::cout << "Creating context for " << (useCPU ? "CPU" : "GPGPU") << "...";
	if(!useCPU){
		try{
			//try gpu
			clever::context_settings settings = clever::context_settings::default_gpu();
			settings.m_profile = true;

			contx = new clever::context(settings);
			std::cout << "success" << std::endl;
		} catch (const std::runtime_error & e){
			//if not use cpu
			clever::context_settings settings = clever::context_settings::default_cpu();
			settings.m_profile = true;
			settings.m_cmd_queue_properties = CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL;

			contx = new clever::context(settings);
			std::cout << "error: fallback on CPU" << std::endl;
		}
	} else {
		clever::context_settings settings = clever::context_settings::default_cpu();
		settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "success" << std::endl;
	}

	wgScan = new WGscan(*contx);
	wgScanStore = new WGscanStore(*contx);
	uniformAdd = new UniformAdd(*contx);

	uint wg = wgScan->krnl.getWorkGroupSize();
	uint maxAlloc = contx->getMaxAllocSize();
	std::cout << "Max work group size is " << wg << " Max mem alloc " << maxAlloc << std::endl;


	unsigned long SIZE = 1024*1024*1024;
	SIZE = min(SIZE, maxAlloc/sizeof(uint));
	clever::vector<uint, 1> input(1, SIZE, *contx);
	std::cout << "Initializing vector with " << SIZE << " values" << std::endl;

	recursiveScan(input.get_mem(), SIZE, wg);

	contx->finish_default_queue();
	std::vector<uint> out(SIZE);
	transfer::download(input, out, *contx, true);

	if(out[SIZE-1] == SIZE-1)
		std::cout << "Prefix sum correct"<< std::endl;
	else {
		std::cout << "Prefix sum INcorrect! Expected: " << SIZE-1 << " Actual: " << out[SIZE-1] << std::endl;

		printVector(out);
	}


}
