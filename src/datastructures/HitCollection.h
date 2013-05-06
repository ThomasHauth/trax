#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <deque>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"
#include "serialize/Event.pb.h"
#include "DetectorGeometry.h"
#include "EventSupplement.h"
#include "LayerSupplement.h"

// todo: can we have a Collection view, which looks like a vector type



#define HIT_COLLECTION_ITEMS GlobalX, GlobalY, GlobalZ, DetectorLayer, DetectorId, HitId, EventNumber


typedef clever::Collection <HIT_COLLECTION_ITEMS> HitCollectionItems;

class HitCollection: public HitCollectionItems
{
public:
	typedef HitCollectionItems dataitems_type;

	typedef std::deque<PB_Event::PHit> tTrackHits; //quickly access both ends of track + random access
	typedef std::map<uint, tTrackHits> tTrackList; //key = trackID, value trackDeque --> easy application of cuts
	typedef tTrackList::value_type tTrackListEntry;

	HitCollection()
	{

	}

	HitCollection(int items) :
			clever::Collection<HIT_COLLECTION_ITEMS>(items)
	{

	}

	// use a range of events to bootstrap the hit collection
	HitCollection(const std::vector < PB_Event::PEvent * >& events) ;

	HitCollection(const PB_Event::PEvent & event) ;

	tTrackList addEvent(const PB_Event::PEvent& event, const DetectorGeometry& geom, EventSupplement & eventSupplement, uint evtInGroup, LayerSupplement & layerSupplement,
			float minPt = 0, int numTracks = -1, bool onlyTracks = false, uint maxLayer = 99) ;

public:
	clever::OpenCLTransfer<HIT_COLLECTION_ITEMS> transfer;

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

class Hit: public clever::CollectionView<HitCollection>
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

	bool operator==(const Hit & rhs) const {
		return getValue<EventNumber>() == rhs.getValue<EventNumber>()
				&& getValue<HitId>() == rhs.getValue<HitId>()
				&& getValue<DetectorId>() == rhs.getValue<DetectorId>()
				&& getValue<DetectorLayer>() == rhs.getValue<DetectorLayer>()
				&& globalX() == rhs.globalX() && globalY() == rhs.globalY() && globalZ() == rhs.globalZ(); //values are only copied ==> should be bitwise identical
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

	float phi() const {
		return atan2(globalY(), globalX());
	}

	float theta() const {
		return atan2( sqrt(globalX()*globalX() + globalY()*globalY()) ,globalZ()*globalZ());
	}

};
