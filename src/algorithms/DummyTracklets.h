#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>

class DummyTracklets : private boost::noncopyable
{
public:
	void run( HitCollectionTransfer & hits )
	{

	}

	KERNEL2_CLASS( dummy_tracklets, cl_mem, double ,
			__kernel void dummy_tracklets( __global double * a, const double b )
	{
		a[ get_global_id( 0 ) ] += b;
	});

};
