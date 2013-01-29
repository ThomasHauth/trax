#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/TrackletCollection.h>

using namespace clever;
using namespace std;

/*
 This class contains the infrastructure and kernel to compute the connectivity quantity for
 triplets. During this process the compatible tracklets which can form a complete track are
 counted and the number is stored with the triplet.

 Input:
 - buffer A holding Triplets to compute : read / write
 - buffer B holding Tripltes to check for connectivity ( can be the same as above ) : read only
 - range to start and end the connectivity search on buffer B

 possible todos:
 - improve performance by using local caching of the connectivity quantity
 - more complex tests ( fitted trajectory, theta / phi values of triptles )

 */
class TripletConnectivity: private boost::noncopyable
{
private:

	clever::context & m_ctx;

public:

	TripletConnectivity(clever::context & ctext);

	static std::string KERNEL_COMPUTE_EVT()
	{
		return "TripletConnectiviy_COMPUTE";
	}
	static std::string KERNEL_STORE_EVT()
	{
		return "TripletConnectiviy_STORE";
	}

	/*
	 by default, only the outermost hits of both triplets are compared for compatibility.
	 The tightPacking option can be set to true to compute the connectivity for overlapping
	 triplets, meaning that two hits must be shared by the triplets

	 */
	void run(TrackletCollectionTransfer const& trackletsBase,
			TrackletCollectionTransfer & trackletsFollowing,
			bool iterateBackwards = false, bool tightPacking = false) const;

	KERNEL6_CLASS( tripletConnectivityWide, cl_mem, cl_mem, cl_mem, cl_mem, cl_uint, cl_uint,
			__kernel void tripletConnectivityWide(
					// tracklet base ( hit id, connectivity )
					__global const uint * tripletBaseHit1,
					__global const uint * tripletBaseCon,
					// tracklet following ( hit id, hit id, connectivity )
					__global uint * tripletFollowHit1,
					__global uint * tripletFollowCon,
					const uint followFirst, const uint followLast
			)
			{
				const size_t gid = get_global_id( 0 );
				const size_t lid = get_local_id( 0 );
				const size_t threads = get_global_size( 0 );

				for ( size_t i = followFirst; i <= followLast; i++ )
				{
					//printf("Doing %i\n" , gid );
					const bool connected = ( tripletBaseHit1[ gid ] == tripletFollowHit1[ i ] );
					//printf("  id1 : %i  id2 : %i\n", tripletBaseHit1[ gid ], tripletFollowHit1[ i ]);
					//printf("  is connected %i\n" , connected );
					//printf("  connectivity old is %i\n" , tripletFollowCon [ i ] );

					//if ( connected )
					//{
					//	atomic_add( tripletFollowCon + i , tripletFollowCon[ i ] + 1 );
					//}

					// this is the branch-less version, test the performance
					// the following triplet inherits the connectivity from the base

					tripletFollowCon[i] = connected * (tripletBaseCon[ gid ] + 1);

					//if ( connected )
					//{
					//	atomic_add( tripletFollowCon + i , ( tripletBaseCon[ gid ] + 1 ) );
					//}

					//printf("  connectivity is now %i\n" , tripletFollowCon [ i ] );
				}

			})
	;

	KERNEL8_CLASS( tripletConnectivityTight, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_uint, cl_uint,
			__kernel void tripletConnectivityTight(
					// tracklet base ( hit id, hit id, connectivity )
					__global uint const* tripletBaseHit1,
					__global uint const* tripletBaseHit2,
					__global uint const* tripletBaseCon,

					// tracklet following ( hit id, hit id, connectivity )
					__global uint const* tripletFollowHit1,
					__global uint const* tripletFollowHit2,
					__global uint * tripletFollowCon,

					const uint followFirst, const uint followLast
			)
			{
				const size_t gid = get_global_id( 0 );
				const size_t lid = get_local_id( 0 );
				const size_t threads = get_global_size( 0 );

				for ( size_t i = followFirst; i <= followLast; i++ )
				{
					//printf("Doing %i\n" , gid );
					// the comparison has to be made criss / cross
					const bool connected = ( tripletBaseHit1[ gid ] == tripletFollowHit1[ i ] ) &&
					( tripletBaseHit2[ gid ] == tripletFollowHit2[ i ] );

					//printf("  id1 : %i  f_id1 : %i\n", tripletBaseHit1[ gid ], tripletFollowHit1[ i ]);
					//printf("  id2 : %i  f_id2 : %i\n", tripletBaseHit2[ gid ], tripletFollowHit2[ i ]);

					//printf("  is connected %i\n" , connected );
					//printf("  connectivity old is %i\n" , tripletFollowCon [ i ] );

					//if ( connected )
					//{
					//	atomic_add( tripletFollowCon + i , tripletFollowCon[ i ] + 1 );
					//}

					// this is the branch-less version, test the performance
					// the following triplet inherits the connectivity from the base
					//if ( connected )
					//{
					//	atomic_add( tripletFollowCon + i , ( tripletBaseCon[ gid ] + 1 ) );
					//}

					tripletFollowCon[i] = connected * (tripletBaseCon[ gid ] + 1);

					//printf("  connectivity is now %i\n" , tripletFollowCon [ i ] );
				}

			})
	;

};
