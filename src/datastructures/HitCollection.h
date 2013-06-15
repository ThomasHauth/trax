#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <deque>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include <datastructures/CommonTypes.h>
#include <datastructures/serialize/Event.pb.h>
#include <datastructures/DetectorGeometry.h>
#include <datastructures/EventSupplement.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Logger.h>
#include <datastructures/TripletConfiguration.h>

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
			const TripletConfigurations & layerTriplets, int numTracks = -1, bool onlyTracks = false) ;

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

		x = getValue<GlobalX>();
		y = getValue<GlobalY>();
		z = getValue<GlobalZ>();

	}

	// read-only get a pointer to one hit in the collection
	Hit(const HitCollection & collection, index_type i) :
		clever::CollectionView<HitCollection>(collection, i)
		{

		x = getValue<GlobalX>();
		y = getValue<GlobalY>();
		z = getValue<GlobalZ>();

		}

// create a new hit in the collection
	Hit(HitCollection & collection) :
			clever::CollectionView<HitCollection>(collection)
	{
		x = 0;
		y = 0;
		z = 0;
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
		return x;
	}

	float globalY() const
	{
		return y;
	}

	float globalZ() const
	{
		return z;
	}

	float phi() const {
		return atan2(globalY(), globalX());
	}

	float theta() const {
		return atan2( sqrt(globalX()*globalX() + globalY()*globalY()) ,globalZ()*globalZ());
	}

protected:
	float x;
	float y;
	float z;

};

class PHitWrapper : public Hit {

public:

	PHitWrapper(const PB_Event::PHit & hit)
	: Hit(pHitCollection)
	{
		x = hit.position().x();
		y = hit.position().y();
		z = hit.position().z();
	}

private:
	static HitCollection pHitCollection;

};
