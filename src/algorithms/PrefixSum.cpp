#include "PrefixSum.h"
#include <iomanip>

const std::string PrefixSum::oclDEFINES = 	 "#define MEMORY_BANK_COUNT       (16) \n "
											 "#define LOG2_MEMORY_BANK_COUNT   (4) \n "
											 "#define MEMORY_BANK_OFFSET(index) ((index) >> LOG2_MEMORY_BANK_COUNT)";

cl_event PrefixSum::run(cl_mem input, const uint size, uint nThreads)
{

	VLOG << "scanning " << size << " elements with max " << nThreads << " work-items ";
	createPartialSums(size, nThreads);
	cl_event evt = recursiveScan(input, size, nThreads, 0);

	return evt;
}

cl_event PrefixSum::run(cl_mem input, const uint size, uint nThreads, std::vector<cl_event> & lEvents)
{

	VLOG << "scanning " << size << " elements with max " << nThreads << " work-items ";
	createPartialSums(size, nThreads);

	uint i = PrefixSum::events.size();

	cl_event evt = recursiveScan(input, size, nThreads, 0);

	for(; i < PrefixSum::events.size(); ++i) //add all events generated in recursive scan
		lEvents.push_back(PrefixSum::events[i]);

	return evt;
}


void PrefixSum::createPartialSums(uint size, uint wg){

	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));
	uint level = 0;
	while(nGroups > 1){
		PLOG << "Creating partial sum buffer for " << nGroups << " entries in level " << level++ << std::endl;

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

	PLOG << "Recursively scanning " << size << " elements with max "
			<< wg << " work-items ";

	// nGroups = size/(2 * workGroupSize)
	uint nGroups = (uint) std::max(1.0f, ceil(((float) size)/(wg<<1)));
	uint padding = (wg<<1) / MEMORY_BANK_COUNT;

	PLOG << "in " << nGroups << " work-groups" << std::endl;

	if(nGroups == 1){
		//only one work-group needed, base case
		uint localSize = std::max(1.0f, ceil(((float) size)/2));
		localSize = getNextPowerOfTwo(localSize);


		PLOG << "Base case: global size = local size: " << localSize << std::endl;

		cl_event evt = prefixSumWG.run(input, size,
			local_param(sizeof(cl_uint), (wg<<1)+padding),
			range(localSize), range(localSize) );

		PrefixSum::events.push_back(evt);

		return evt;
	} else {
		//scan with sums into partial
		PLOG << "Recursive case: allocating " << nGroups << " partial sums ";
		PLOG << "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;

		cl_event evt = prefixSumPartial.run(input, size, partialSums[level]->get_mem(),
				local_param(sizeof(cl_uint), (wg<<1)+padding),
				range(wg*nGroups), range(wg) );

		PrefixSum::events.push_back(evt);

		if(PROLIX){
			ctx.finish_default_queue();

			std::vector<uint> lPartial(nGroups);
			transfer::download(*partialSums[level], lPartial,ctx, true);
			printVector(lPartial);
		}

		recursiveScan(partialSums[level]->get_mem(), nGroups, wg, level+1);

		if(PROLIX){
			ctx.finish_default_queue();

			std::vector<uint> lPartial(nGroups);
			transfer::download(*partialSums[level], lPartial,ctx, true);
			printVector(lPartial);
		}

		PLOG << "Adding " << nGroups << " partial sums to " << size << " inputs "
				<< "global size: " << (wg*nGroups) << " local size: " << wg << std::endl;

		evt = prefixSumUniformAdd.run(input, size, partialSums[level]->get_mem(),
				range(wg*nGroups), range(wg));

		PrefixSum::events.push_back(evt);

		return evt;
	}

}

void PrefixSum::printVector(std::vector<uint> in){

	for(uint i = 0; i < in.size(); ++i){
		PLOG <<  std::setw(4) << std::setfill(' ') << i << ":"
				<< std::setw(4) << std::setfill(' ') << in[i] << "   ";
		//add empty lines with prefix sum is broken
		if(i < in.size()-1 && in[i] > in[i+1])
			PLOG << std::endl << std::endl;
	}

	PLOG << std::endl;
}
