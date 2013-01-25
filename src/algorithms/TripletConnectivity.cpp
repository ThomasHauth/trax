#include "TripletConnectivity.h"

#include <algorithm>

TripletConnectivity::TripletConnectivity(clever::context & ctext) :
		m_ctx(ctext), tripletConnectivity(ctext)
{
	// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
	std::cout << "tripletConnectivity WorkGroupSize: "
			<< tripletConnectivity.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

void TripletConnectivity::run(
		TrackletCollectionTransfer const& trackletsInitial,
		TrackletCollectionTransfer & trackletsFollowing, bool iterateBackwards) const
{
	cl_mem intialHits;
	cl_mem followingHits;

	if (! iterateBackwards)
	{
		intialHits = trackletsInitial.buffer(TrackletHit3());
		followingHits = trackletsFollowing.buffer(TrackletHit1());
	}
	else
	{
		intialHits = trackletsInitial.buffer(TrackletHit1());
		followingHits = trackletsFollowing.buffer(TrackletHit3());
	}

	tripletConnectivity.run(intialHits,
			trackletsInitial.buffer(TrackletConnectivity()), followingHits,
			trackletsFollowing.buffer(TrackletConnectivity()), 0,
			// be sure to not produce -1 ( >> unsigned ) if there are not triplets
			std::max((unsigned int) 0,
					(unsigned int) trackletsFollowing.defaultRange().getSize()
							- 1), trackletsInitial.defaultRange());
	/*
	 clever::vector<uint2, 1> * m_pairs = generateAllPairs(hits, nThreads,
	 layers, layerSupplement);
	 //PairGeneratorSector pairGen(ctx);
	 //clever::vector<uint2,1> * m_pairs = pairGen.run(hits, nThreads, layers, layerSupplement , nSectors);

	 //clever::vector<uint2,1> * m_triplets = generateAllTriplets(hits, nThreads, layers, hitCount, 1.2*dThetaCut, 1.2*dPhiCut, *m_pairs);
	 TripletThetaPhiPredictor predictor(ctx);
	 float dThetaWindow = 0.1;
	 float dPhiWindow = 0.1;
	 clever::vector<uint2, 1> * m_triplets = predictor.run(hits, geom, dict,
	 nThreads, layers, layerSupplement, dThetaWindow, dPhiWindow,
	 *m_pairs);
	 int nTripletCandidates = m_triplets->get_count();

	 std::cout << "Initializing oracle...";
	 clever::vector<uint, 1> m_oracle(0,
	 std::ceil(nTripletCandidates / 32.0), ctx);
	 std::cout << "done[" << m_oracle.get_count() << "]" << std::endl;

	 std::cout << "Initializing prefix sum...";
	 clever::vector<uint, 1> m_prefixSum(0, nThreads + 1, ctx);
	 std::cout << "done[" << m_prefixSum.get_count() << "]" << std::endl;

	 std::cout << "Running filter kernel...";
	 cl_event evt = tripletThetaPhiCheck.run(
	 //configuration
	 dThetaCut, dPhiCut, nTripletCandidates,
	 // input
	 m_pairs->get_mem(), m_triplets->get_mem(),
	 hits.buffer(GlobalX()), hits.buffer(GlobalY()),
	 hits.buffer(GlobalZ()),
	 // output
	 m_oracle.get_mem(), m_prefixSum.get_mem(),
	 //thread config
	 nThreads);
	 std::cout << "done" << std::endl;

	 ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

	 std::cout << "Fetching prefix sum...";
	 std::vector < uint > prefixSum(m_prefixSum.get_count());
	 transfer::download(m_prefixSum, prefixSum, ctx);
	 std::cout << "done" << std::endl;
	 */

}
