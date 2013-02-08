#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class AlmostBruteForceTracklets: private boost::noncopyable
{
private:
	clever::scalar<float> m_dThetaCut;
	clever::scalar<float> m_dPhiCut;
	clever::scalar<float> m_tipCut;

public:
	clever::scalar<unsigned int> m_trackletsFound;

	AlmostBruteForceTracklets(clever::context & ctext) :
		m_dThetaCut(ctext),
		m_dPhiCut(ctext),
		m_tipCut(ctext),
		m_trackletsFound(ctext),
		almostBruteForce_tracklets(ctext)
	{
		// create the buffers this algorithm will need to run
	}

	unsigned int run(HitCollectionTransfer & hits,
			TrackletCollectionTransfer& tracklets)
	{

		m_trackletsFound.fromVariable(0);

		//cuts
		m_dThetaCut.fromVariable(0.01);
		m_dPhiCut.fromVariable(0.1);
		m_tipCut.fromVariable(1);

		almostBruteForce_tracklets.run(
				//configuration
				m_dThetaCut.get_mem(), m_dPhiCut.get_mem(), m_tipCut.get_mem(),
				// input
				hits.buffer(GlobalX()), hits.buffer(GlobalY()), hits.buffer(GlobalZ()),
				hits.buffer(HitId()),
				// interemediate data
				m_trackletsFound.get_mem(),
				// output
				tracklets.buffer(TrackletHit1()),
				tracklets.buffer(TrackletHit2()),
				tracklets.buffer(TrackletHit3()),
				tracklets.buffer(TrackletId()), hits.defaultRange());

		std::cout << "Found " << m_trackletsFound.getValue() << " tracklets" << std::endl;
		return m_trackletsFound.getValue();
	}

	KERNEL12_CLASS( almostBruteForce_tracklets, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem , cl_mem, cl_mem,
			__kernel void almostBruteForce_tracklets(
					//configuration
					__global const float * dThetaCut, __global const float * dPhiCut, __global const float * tipCut,
					// hit input
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ, __global const uint * hitId,
					// intermeditate data: tracklet count, concurrent access using atomics
					__global uint * trackletsFound,
					// output of tracklet data
					__global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3, __global uint * trackletId )
			{
				size_t gId = get_global_id( 0 );
				size_t gSize = get_global_size( 0 );
				//printf ( "Running dummy_tracklets_cluster on HitId (%i)\n", hitId[ gId ] );

				size_t firstHit = gId;

				for ( size_t secondHit = 0; secondHit < gSize; secondHit ++ )
				{
					for ( size_t thirdHit = 0; thirdHit < gSize; thirdHit ++ )
					{
						//tanTheta1
						float angle1 = sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit])*(hitGlobalX[secondHit] - hitGlobalX[firstHit])
								+ (hitGlobalY[secondHit] - hitGlobalY[firstHit])*(hitGlobalY[secondHit] - hitGlobalY[firstHit]))
														/ ( hitGlobalZ[secondHit] - hitGlobalZ[firstHit] );
						//tanTheta2
						float angle2 = sqrt((hitGlobalX[thirdHit] - hitGlobalX[secondHit])*(hitGlobalX[thirdHit] - hitGlobalX[secondHit])
								+ (hitGlobalY[thirdHit] - hitGlobalY[secondHit])*(hitGlobalY[thirdHit] - hitGlobalY[secondHit]))
																				/ ( hitGlobalZ[thirdHit] - hitGlobalZ[secondHit] );
						float ratio = angle2/angle1;
						if(!(1-*dThetaCut <= ratio && ratio <= 1+*dThetaCut))
							continue; //this is really bad!

						//tanPhi1
						angle1 = (hitGlobalY[secondHit] - hitGlobalY[firstHit]) / ( hitGlobalX[secondHit] - hitGlobalX[firstHit] );
						//tanPhi2
						angle2 = (hitGlobalY[thirdHit] - hitGlobalY[secondHit]) / ( hitGlobalX[thirdHit] - hitGlobalX[secondHit] );

						ratio = angle2/angle1;
						if(!(1-*dPhiCut <= ratio && ratio <= 1+*dPhiCut))
							continue; //this is really bad!

						//circle fit
						//map points to parabloid: (x,y) -> (x,y,x^2+y^2)
						float3 pP1 = (float3) (hitGlobalX[firstHit],
								hitGlobalY[firstHit],
								hitGlobalX[firstHit] * hitGlobalX[firstHit] + hitGlobalY[firstHit] * hitGlobalY[firstHit]);

						float3 pP2 = (float3) (hitGlobalX[secondHit],
								hitGlobalY[secondHit],
								hitGlobalX[secondHit] * hitGlobalX[secondHit] + hitGlobalY[secondHit] * hitGlobalY[secondHit]);

						float3 pP3 = (float3) (hitGlobalX[thirdHit],
								hitGlobalY[thirdHit],
								hitGlobalX[thirdHit] * hitGlobalX[thirdHit] + hitGlobalY[thirdHit] * hitGlobalY[thirdHit]);

						//span two vectors

						/// DONT COMMIT commented because not commited in clever right now
						//
						//float3 a = pP2 - pP1;
						//float3 b = pP3 - pP1;
						float3 a;
						float3 b;

						//compute unit cross product
						float3 n;// dito = cross(a,b);
						//n = normalize(n);

						//formula for orign and radius of circle from Strandlie et al.
						float2 cOrigin = (float2) ((-n.x) / (2*n.z),
								(-n.y) / (2*n.z));

						float c = -(n.x*pP1.x + n.y*pP1.y + n.z*pP1.z);

						float cR = sqrt((1 - n.z*n.z - 4 * c * n.z) / (4*n.z*n.z));

						//find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
						float2 v = -cOrigin; v = normalize(v);
						float2 pCA = (float2) (cOrigin.x + cR*v.x,
								cOrigin.y + cR*v.y);

						//TIP = distance of point of closest approach to origin
						float tip = sqrt(pCA.x*pCA.x + pCA.y*pCA.y);

						if(tip <= *tipCut)
							continue; //this is bad

						const uint thisTrackletId = atomic_inc( trackletsFound );

						trackletHitId1[thisTrackletId] = hitId[ firstHit ];
						trackletHitId2[thisTrackletId] = hitId[ secondHit ];
						trackletHitId3[thisTrackletId] = hitId[ thirdHit ];
						trackletId[thisTrackletId] = thisTrackletId;
					}
				}
			});

		};
