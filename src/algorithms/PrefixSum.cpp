#include "PrefixSum.h"

const std::string PrefixSum::oclDEFINES = 	 "#define MEMORY_BANK_COUNT       (16) \n "
											 "#define LOG2_MEMORY_BANK_COUNT   (4) \n "
											 "#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)";

cl_event PrefixSum::run(cl_mem input, const uint size, uint nThreads)
{

	createPartialSums(size, nThreads);
	cl_event evt = recursiveScan(input, size, nThreads, 0);

	return evt;
}

void PrefixSum::createPartialSums(uint size, uint wg){

	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));
	uint level = 0;
	while(nGroups > 1){
#ifdef DEBUG_OUT
		std::cout << "Creating partial sum buffer for " << nGroups << " entries in level " << level++ << std::endl;
#endif

		clever::vector<uint, 1> *partial = new clever::vector<uint, 1>(0, nGroups, ctx);
		partialSums.push_back(partial);
		nGroups = (uint) std::max(1.0f, ceil(((float) nGroups)/(wg<<1)));
	}
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

cl_event PrefixSum::recursiveScan(cl_mem input, uint size, uint wg, uint level){

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

		cl_event evt = prefixSum.run(input, size, 0,
			local_param(sizeof(cl_uint), (wg<<1)+padding),
			range(localSize), range(localSize) );

		return evt;
	} else {
		//scan with sums into partial
#ifdef DEBUG_OUT
		std::cout << "Recursive case: allocating " << nGroups << " partial sums ";
		std::cout << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;
#endif

		prefixSum.run(input, size, partialSums[level]->get_mem(),
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

		cl_event evt = uniformAdd.run(input, size, partialSums[level]->get_mem(),
				range(wg*nGroups), range(wg));

		return evt;
	}

}
