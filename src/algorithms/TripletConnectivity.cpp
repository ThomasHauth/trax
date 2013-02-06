#include "TripletConnectivity.h"

#include <algorithm>

TripletConnectivity::TripletConnectivity(clever::context & ctext) :
		m_ctx(ctext), tripletConnectivityTight(ctext), tripletConnectivityWide(
				ctext)
{
	// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
	std::cout << "tripletConnectivityWide WorkGroupSize: "
			<< tripletConnectivityWide.getWorkGroupSize() << std::endl;
	std::cout << "tripletConnectivityTight WorkGroupSize: "
			<< tripletConnectivityTight.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

void TripletConnectivity::run(
		TrackletCollection  const& trackletsInitial,
		TrackletCollection & trackletsFollowing, bool iterateBackwards,
		bool tightPacking) const
{
	cl_mem intialHits1;
	cl_mem intialHits2;

	cl_mem followingHits1;
	cl_mem followingHits2;

	if (!iterateBackwards)
	{
		intialHits1 = trackletsInitial.transfer.buffer(TrackletHit3());
		followingHits1 = trackletsFollowing.transfer.buffer(TrackletHit1());
	}
	else
	{
		intialHits1 = trackletsInitial.transfer.buffer(TrackletHit1());
		followingHits1 = trackletsFollowing.transfer.buffer(TrackletHit3());
	}

	const size_t maxBaseItem = std::max((unsigned int) 0,
			(unsigned int) trackletsInitial.transfer.defaultRange().getSize() - 1);

	if (!tightPacking)
	{
		tripletConnectivityWide.run(intialHits1,
				trackletsInitial.transfer.buffer(TrackletConnectivity()), followingHits1,
				trackletsFollowing.transfer.buffer(TrackletConnectivity()), 0,
				// be sure to not produce -1 ( >> unsigned ) if there are not triplets
				maxBaseItem, trackletsFollowing.transfer.defaultRange());
	}
	else
	{
		assert(!iterateBackwards); // "not supported"

		// get the other two needed hits
		intialHits2 = intialHits1;
		intialHits1 = trackletsInitial.transfer.buffer(TrackletHit2());

		followingHits2 = trackletsFollowing.transfer.buffer(TrackletHit2());
		// ... and fire away
		tripletConnectivityTight.run(intialHits1, intialHits2,
				trackletsInitial.transfer.buffer(TrackletConnectivity()), followingHits1,
				followingHits2,
				trackletsFollowing.transfer.buffer(TrackletConnectivity()), 0,
				// be sure to not produce -1 ( >> unsigned ) if there are not triplets
				maxBaseItem, trackletsFollowing.transfer.defaultRange());
	}
}
