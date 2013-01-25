#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"
#include "HitCollection.h"

struct TrackletHit1: public clever::UIntItem
{
};
struct TrackletHit2: public clever::UIntItem
{
};
struct TrackletHit3: public clever::UIntItem
{
};

struct TrackletId: public clever::UIntItem
{
};

// todo: can also be uchar
struct TrackletConnectivity: public clever::UIntItem
{
};

#define TRACKLET_COLLECTION_ITEMS TrackletId, TrackletHit1, TrackletHit2, TrackletHit3, TrackletConnectivity

typedef clever::Collection<TRACKLET_COLLECTION_ITEMS> TrackletCollectionItems;

class TrackletCollection: public TrackletCollectionItems
{
public:
	typedef TrackletCollectionItems dataitems_type;

	TrackletCollection()
	{

	}

	TrackletCollection(int items) :
			clever::Collection<TRACKLET_COLLECTION_ITEMS>(items)
	{

	}
};

typedef clever::OpenCLTransfer<TRACKLET_COLLECTION_ITEMS> TrackletCollectionTransfer;

class Tracklet: private clever::CollectionView<TrackletCollection>
{
public:
// get a pointer to one hit in the collection
	Tracklet(TrackletCollection & collection, index_type i) :
			clever::CollectionView<TrackletCollection>(collection, i)
	{

	}

// create a new hit in the collection
	Tracklet(TrackletCollection & collection) :
			clever::CollectionView<TrackletCollection>(collection)
	{
	}

	float hit1() const
	{
		return getValue<TrackletHit1>();
	}

	float hit2() const
	{
		return getValue<TrackletHit2>();
	}

	float hit3() const
	{
		return getValue<TrackletHit3>();
	}

	float id() const {
		return getValue<TrackletId>();
	}

	bool isValid(const HitCollection& hits) const {
		return hits.getValue(HitId(),hit1()) == hits.getValue(HitId(),hit2())
						&& hits.getValue(HitId(),hit1()) == hits.getValue(HitId(),hit3());
	}

	uint trackId(const HitCollection& hits) const {
		if(isValid(hits))
			return hits.getValue(HitId(),hit1());

		return 0;

	}

};

