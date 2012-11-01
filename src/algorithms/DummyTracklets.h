#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>

class DummyTracklets: private boost::noncopyable
{
public:
	DummyTracklets(clever::context & ctext) :
			dummy_tracklets(ctext)
	{

	}

	void run(HitCollectionTransfer & hits)
	{
		dummy_tracklets.run(hits.buffer(GlobalX()), hits.buffer(GlobalY()), hits.buffer(HitId()),
				hits.defaultRange());
	}

	KERNEL3_CLASS( dummy_tracklets, cl_mem, cl_mem , cl_mem,
			__kernel void dummy_tracklets( __global float * hitGlobalX, __global float * hitGlobalY, __global uint * hitId )
			{
				size_t gId = get_global_id( 0 );
				printf ( " Running dummy_tracklets_cluster on HitId (%i)\n", hitId[ gId ] );
			});

		};
