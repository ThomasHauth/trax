#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"

// todo: can we have a Collection view, which looks like a vector type



#define HIT_COLLECTION_ITEMS GlobalX, GlobalY, GlobalZ, DetectorId, HitId


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

};

