#include <boost/noncopyable.hpp>
#include <clever/clever.hpp>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

using namespace clever;
using namespace std;

#define MEMORY_BANK_COUNT       (16)  // Adjust to your architecture
#define LOG2_MEMORY_BANK_COUNT   (4)  // Set to log2(MEMORY_BANK_COUNT)
#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)

std::string oclDEFINES = "#define MEMORY_BANK_COUNT       (16) \n "
						 "#define LOG2_MEMORY_BANK_COUNT   (4) \n "
						 "#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)";

////////////////////////////////////////////////////////////////////////////////////////////////////

class WGscan{

public:
	WGscan(clever::context & ctext  )
		: krnl ( ctext ){ }


	KERNEL3_CLASSP( krnl, cl_mem, uint, local_param, oclDEFINES,

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
		if(gOffset + lid + ls < size)
			input[gOffset + lid + ls] = data[lid+ls + MEMORY_BANK_OFFSET(lid+ls)];
		barrier(CLK_LOCAL_MEM_FENCE);

	}
	);
};

class WGscanStore{

public:
	WGscanStore(clever::context & ctext  )
		: krnl ( ctext ){ }


	KERNEL4_CLASSP( krnl, cl_mem, uint, cl_mem, local_param, oclDEFINES,

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
		if(gOffset + lid < size)
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
std::vector<clever::vector<uint, 1> *> partialSums;

//////////////////////////////////////////////////

//#define DEBUG_OUT

void printVector(std::vector<uint> in){

#ifdef DEBUG_OUT
	for(uint i = 0; i < in.size(); ++i){
		std::cout <<  std::setw(4) << std::setfill(' ') << i << ":"
				<< std::setw(4) << std::setfill(' ') << in[i] << "   ";
		//add empty lines with prefix sum is broken
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

uint64_t
GetCurrentTime()
{
	struct timeval t;
	gettimeofday(&t, NULL);

    return t.tv_sec * 1E6 + t.tv_usec;
}

double
SubtractTimeInSec( uint64_t endtime, uint64_t starttime )
{
	uint64_t difference = endtime - starttime;

	return difference * 1E-6;
}

/////////////////////////////////////////////////////////////////////////////////

void createPartialSums(uint size, uint wg){

	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));
	uint level = 0;
	while(nGroups > 1){
		std::cout << "Creating partial sum buffer for " << nGroups << " entries in level " << level++ << std::endl;
		clever::vector<uint, 1> *partial = new clever::vector<uint, 1>(0, nGroups, *contx);
		partialSums.push_back(partial);
		nGroups = (uint) std::max(1.0f, ceil(((float) nGroups)/(wg<<1)));
	}
}


//#define DEBUG_OUT
void recursiveScan(cl_mem input, uint size, uint wg, uint level){

#ifdef DEBUG_OUT
	std::cout << "Recursively scanning " << size << " elements with max "
			  << wg << " work-items ";
#endif

	// nGroups = size/(2 * workGroupSize)
	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));
	uint padding = (wg<<1) / MEMORY_BANK_COUNT;

#ifdef DEBUG_OUT
	std::cout << "in " << nGroups << " work-groups" << std::endl;
#endif

	if(nGroups == 1){
		//only one work-group needed, base case
		uint localSize = std::max(1.0f, ceil(((float) size)/2));
		localSize = getNextPowerOfTwo(localSize);

#ifdef DEBUG_OUT
		std::cout << "Base case: global size = local size: " << localSize << std::endl;
#endif

		wgScan->krnl.run(input, size,
				local_param(sizeof(cl_uint), (wg<<1)+padding),
				range(localSize), range(localSize) );

		return;
	} else {
		//scan with sums into partial
#ifdef DEBUG_OUT
		std::cout << "Recursive case: allocating " << nGroups << " partial sums ";
		std::cout << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;
#endif

		wgScanStore->krnl.run(input, size, partialSums[level]->get_mem(),
				local_param(sizeof(cl_uint), (wg<<1)+padding),
				range(wg*nGroups), range(wg) );

#ifdef DEBUG_OUT
		contx->finish_default_queue();

		std::vector<uint> lPartial(nGroups);
		transfer::download(*partialSums[level], lPartial,*contx, true);
		printVector(lPartial);
#endif

		recursiveScan(partialSums[level]->get_mem(), nGroups, wg, level+1);

#ifdef DEBUG_OUT
		contx->finish_default_queue();

		transfer::download(*partialSums[level], lPartial,*contx, true);
		printVector(lPartial);

		std::cout << "Adding " << nGroups << " partial sums to " << size << " inputs "
				  << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;

#endif

		uniformAdd->krnl.run(input, size, partialSums[level]->get_mem(),
				range(wg*nGroups), range(wg));
	}

}

int main(int argc, char **argv) {

	bool useCPU = false;

	uint ITERATIONS = 1000;

	std::cout << "Creating context for " << (useCPU ? "CPU" : "GPGPU") << "...";
	if(!useCPU){
		try{
			//try gpu
			clever::context_settings settings = clever::context_settings::default_gpu();
			//settings.m_profile = true;

			contx = new clever::context(settings);
			std::cout << "success" << std::endl;
		} catch (const std::runtime_error & e){
			//if not use cpu
			clever::context_settings settings = clever::context_settings::default_cpu();
			//settings.m_profile = true;
			//settings.m_cmd_queue_properties = CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL;

			contx = new clever::context(settings);
			std::cout << "error: fallback on CPU" << std::endl;
		}
	} else {
		clever::context_settings settings = clever::context_settings::default_cpu();
		//	settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "success" << std::endl;
	}

	wgScan = new WGscan(*contx);
	wgScanStore = new WGscanStore(*contx);
	uniformAdd = new UniformAdd(*contx);

	uint WG = 1024;
	uint wg = wgScan->krnl.getWorkGroupSize();
	wg = min(WG, wg);
	uint maxAlloc = contx->getMaxAllocSize();
	std::cout << "Max work group size is " << wg << " Max mem alloc " << maxAlloc << std::endl;


	unsigned long SIZE = 1048713;
	SIZE = min(SIZE, maxAlloc/sizeof(uint));
	clever::vector<uint, 1> input(1, SIZE, *contx);
	std::cout << "Initializing vector with " << SIZE << " values" << std::endl;

	createPartialSums(SIZE,wg);
	recursiveScan(input.get_mem(), SIZE, wg, 0);

	std::vector<uint> out(SIZE);
	transfer::download(input, out, *contx, true);
	contx->finish_default_queue();

	uint64_t t0 = GetCurrentTime();
	for (uint i = 0; i < ITERATIONS; i++)
	{
		recursiveScan(input.get_mem(), SIZE, wg, 0);
	}
	contx->finish_default_queue();
	uint64_t t1 = GetCurrentTime();

	double t = SubtractTimeInSec(t1, t0);
	printf("Exec Time:  %.2f ms\n", 1000.0 * t / (double)(ITERATIONS));
	printf("Throughput: %.2f GB/sec\n", 1e-9 * SIZE * sizeof(uint) * ITERATIONS / t);

	if(out[SIZE-1] == SIZE-1)
		std::cout << "Prefix sum correct"<< std::endl;
	else {
		std::cout << "Prefix sum INcorrect! Expected: " << SIZE-1 << " Actual: " << out[SIZE-1] << std::endl;

		printVector(out);
	}


}
