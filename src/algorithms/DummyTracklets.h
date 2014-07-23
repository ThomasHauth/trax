#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class DummyTracklets: private boost::noncopyable
{
public:
	clever::scalar<unsigned int> m_trackletsFound;

	DummyTracklets(clever::context & ctext) :
			m_trackletsFound(ctext),
			dummy_tracklets(ctext)
	{
		// create the buffers this algorithm will need to run
	}

	unsigned int run(HitCollectionTransfer & hits,
			TrackletCollectionTransfer& tracklets)
	{

		m_trackletsFound.fromVariable(0);
		dummy_tracklets.run(
				// input
				hits.buffer(GlobalX()), hits.buffer(GlobalY()),
				hits.buffer(HitId()),
				// interemediate data
				m_trackletsFound.get_mem(),
				// output
				tracklets.buffer(TrackletHit1()),
				tracklets.buffer(TrackletHit1()),
				tracklets.buffer(TrackletHit1()),
				tracklets.buffer(TrackletId()), hits.defaultRange());

		std::cout << "Found " << m_trackletsFound.getValue() << " tracklets" << std::endl;
		return m_trackletsFound.getValue();
	}

	KERNEL_CLASS( dummy_tracklets,
			__kernel void dummy_tracklets(
					// hit input
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const uint * hitId,
					// intermeditate data: tracklet count, concurrent access using atomics
					__global uint * trackletsFound,
					// output of tracklet data
					__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3, __global uint * trackletId )
			{
				size_t gId = get_global_id( 0 );
				size_t gSize = get_global_size( 0 );
				//printf ( "Running dummy_tracklets_cluster on HitId (%i)\n", hitId[ gId ] );

				for ( size_t secondHit = 0; secondHit < gSize; secondHit ++ )
				{
					for ( size_t thirdHit = 0; thirdHit < gSize; thirdHit ++ )
					{
						// kill the obvious no-brainers
						if ( ( gId == secondHit  ) || ( gId == thirdHit ) )
							continue;
						if (secondHit == thirdHit )
							continue;

						const uint thisTrackletId = atomic_inc( trackletsFound );

						trackletHitId1[thisTrackletId] = hitId[ gId ];
						trackletHitId2[thisTrackletId] = hitId[ secondHit ];
						trackletHitId3[thisTrackletId] = hitId[ thirdHit ];
						trackletId[thisTrackletId] = thisTrackletId;
					}
				}
			},
		cl_mem, cl_mem , cl_mem, cl_mem, cl_mem, cl_mem , cl_mem, cl_mem
	);

};
