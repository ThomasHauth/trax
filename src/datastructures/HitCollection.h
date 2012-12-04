#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <deque>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"
#include "serialize/Event.pb.h"

// todo: can we have a Collection view, which looks like a vector type



#define HIT_COLLECTION_ITEMS GlobalX, GlobalY, GlobalZ, DetectorLayer, DetectorId, HitId, EventNumber


typedef clever::Collection <HIT_COLLECTION_ITEMS> HitCollectiontems;

class HitCollection: public HitCollectiontems
{
public:
	typedef HitCollectiontems dataitems_type;

	HitCollection()
	{

	}

	HitCollection(int items) :
			clever::Collection<HIT_COLLECTION_ITEMS>(items)
	{

	}

	// use a range of events to bootstrap the hit collection
	HitCollection(const std::vector < PB_Event::PEvent * >& events ) ;

	HitCollection(const PB_Event::PEvent & event) ;

	void addEvent(const PB_Event::PEvent& event, float minPt = 0, int numTracks = -1, bool onlyTracks = false, uint maxLayer = 99);

private:
	typedef std::deque<PB_Event::PHit> tTrackHits; //quickly access both ends of track + random access
	typedef std::map<uint, tTrackHits> tTrackList; //key = trackID, value trackDeque --> easy application of cuts
	typedef tTrackList::value_type tTrackListEntry;
};
/*
class GlobalPosition: private CollectionView<HitCollection>
{
public:
	// get a pointer to one hit in the collection
	GlobalPosition(HitCollection & collection, index_type i) :
			CollectionView<HitCollection>(collection, i)
	{

	}

	float x() const
	{
		return getValue<GlobalX>();
	}

	float y() const
	{
		return getValue<GlobalY>();
	}

	void setX(float v)
	{
		setValue<GlobalX>(v);
	}

	void setY(float v)
	{
		setValue<GlobalY>(v);
	}

	// * a bit difficult, where do we store to ?
	 GlobalPosition & GlobalPosition::operator+=(const GlobalPosition &rhs) {
	 rs
	 }
};
*/
typedef clever::OpenCLTransfer<HIT_COLLECTION_ITEMS> HitCollectionTransfer;

class Hit: private clever::CollectionView<HitCollection>
{
public:
// get a pointer to one hit in the collection
	Hit(HitCollection & collection, index_type i) :
			clever::CollectionView<HitCollection>(collection, i)
	{

	}

// create a new hit in the collection
	Hit(HitCollection & collection) :
			clever::CollectionView<HitCollection>(collection)
	{
	}

	float globalX() const
	{
		return getValue<GlobalX>();
	}

	float globalY() const
	{
		return getValue<GlobalY>();
	}

	float globalZ() const
	{
		return getValue<GlobalZ>();
	}

};

void HitCollection::addEvent(const PB_Event::PEvent& event, float minPt, int numTracks, bool onlyTracks, uint maxLayer) {

	tTrackList tracks;
	for (auto& hit : event.hits()) {

		if(onlyTracks && hit.simtrackid() == 0)
			continue;

		if (hit.simtrackpt() < minPt)
			continue;

		tracks[hit.simtrackid()].push_back(hit);
	}

	int inc = (numTracks != -1) ? tracks.size() / numTracks : 1;
	for(tTrackList::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); std::advance(itTrack, inc)){
		for (auto& hit : itTrack->second) {
			if(hit.layer() <= maxLayer){
				addWithValue(hit.position().x(), hit.position().y(),
						hit.position().z(), hit.layer(), hit.detectorid(),
						hit.hitid(), event.eventnumber());
				/*std::cout << hit.position().x()<< " | " <<  hit.position().y()<< " | " <<  hit.position().z()<< " | " <<
				 hit.layer()<< " | " << hit.detectorid()<< " | " << hit.hitid()<< " | " <<
				 event.eventnumber() << " | " << hit.simtrackid() << " | " << hit.simtrackpt() << std::endl;*/
			}
		}
	}
}
