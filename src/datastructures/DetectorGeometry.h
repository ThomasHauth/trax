#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"
#include "Dictionary.h"

struct RadiusDict: public clever::UCharDictionaryItem
{
};

#define DETECTOR_GEOMETRY_ITEMS DetectorId, DetectorLayer, RadiusDict

typedef clever::Collection<DETECTOR_GEOMETRY_ITEMS> DetectorGeometryItems;

class DetectorGeometry: public DetectorGeometryItems
{
public:
	typedef DetectorGeometryItems dataitems_type;

	DetectorGeometry()
	{

	}

	DetectorGeometry(int items) :
			clever::Collection<DETECTOR_GEOMETRY_ITEMS>(items)
	{

	}

	int resolveDetId(uint detId) const;

public:
	clever::OpenCLTransfer<DETECTOR_GEOMETRY_ITEMS> transfer;
};

class DetUnit: private clever::CollectionView<DetectorGeometry>
{
public:
	// get a pointer to one hit in the collection
	DetUnit(DetectorGeometry & collection, index_type i) :
		clever::CollectionView<DetectorGeometry>(collection, i)
	{

	}

	// get a pointer to one hit in the collection (readonly)
	DetUnit(const DetectorGeometry & collection, index_type i) :
		clever::CollectionView<DetectorGeometry>(collection, i)
	{

	}

	// create a new hit in the collection
	DetUnit(DetectorGeometry & collection) :
		clever::CollectionView<DetectorGeometry>(collection)
	{

	}

	uint detId() const
	{
		return getValue<DetectorId>();
	}

	clever::uchar radiusDict() const
	{
		return getValue<RadiusDict>();
	}

	float radius(Dictionary dict) const
	{
		return DictionaryEntry(dict, radiusDict()).radius();
	}

};

