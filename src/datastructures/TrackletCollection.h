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

	TrackletCollection(int items, uint nEvents, uint nLayerTriplets, clever::context & ctext) :
			clever::Collection<TRACKLET_COLLECTION_ITEMS>(items),
			trackletOffsets(0 , nEvents*nLayerTriplets+1, ctext),
			ctx(ctext),
			lTrackletOffsets(0)
	{

	}

	const std::vector<uint> & getTrackletOffsets() const {
		if(lTrackletOffsets != NULL)
			return *lTrackletOffsets;
		else {
			lTrackletOffsets = new std::vector<uint>(trackletOffsets.get_count());
			clever::transfer::download(trackletOffsets, *lTrackletOffsets, ctx);

			return *lTrackletOffsets;
		}
	}

	void invalidate(){
		delete lTrackletOffsets; lTrackletOffsets = NULL;
	}

	~TrackletCollection(){
		delete lTrackletOffsets;
	}


public:
	clever::OpenCLTransfer<TRACKLET_COLLECTION_ITEMS> transfer;
	clever::vector<uint, 1> trackletOffsets;

private:
	clever::context & ctx;
	mutable std::vector<uint> * lTrackletOffsets;
};

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

	uint hit1() const
	{
		return getValue<TrackletHit1>();
	}

	uint hit2() const
	{
		return getValue<TrackletHit2>();
	}

	uint hit3() const
	{
		return getValue<TrackletHit3>();
	}

	uint id() const {
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

