#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"

struct MinRadius: public clever::FloatItem
{

};

struct MaxRadius: public clever::FloatItem
{
};

#define GEOMETRY_SUPPLEMENT_ITEMS DetectorLayer, MinRadius, MaxRadius

typedef clever::Collection<GEOMETRY_SUPPLEMENT_ITEMS> GeometrySupplementItems;

class GeometrySupplement: public GeometrySupplementItems
{
public:
	typedef GeometrySupplementItems dataitems_type;

	GeometrySupplement()
	{

	}

	GeometrySupplement(int items) :
			clever::Collection<GEOMETRY_SUPPLEMENT_ITEMS>(items)
	{

	}

public:
	clever::OpenCLTransfer<GEOMETRY_SUPPLEMENT_ITEMS> transfer;
};

class LayerGeometry: private clever::CollectionView<GeometrySupplement>
{
public:
	// get a pointer to one hit in the collection
	LayerGeometry(GeometrySupplement & collection, index_type i) :
		clever::CollectionView<GeometrySupplement>(collection, i-1), mLayer(i)
	{

	}

	// get a pointer to one hit in the collection (readonly)
	LayerGeometry(const GeometrySupplement & collection, index_type i) :
		clever::CollectionView<GeometrySupplement>(collection, i-1), mLayer(i)
	{

	}

	uint layer() const
	{
		return mLayer;
	}


	float minRadius() const
	{
		return getValue<MinRadius>();
	}

	float maxRadius() const
	{
		return getValue<MaxRadius>();
	}

private:
	uint mLayer;

};

