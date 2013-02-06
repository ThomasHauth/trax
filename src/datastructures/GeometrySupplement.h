#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"

struct MaxRadius: public clever::FloatItem
{
};

#define GEOMETRY_SUPPLEMENT_ITEMS DetectorLayer, MaxRadius

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

	int resolveDetId(uint detId) const;

public:
	clever::OpenCLTransfer<GEOMETRY_SUPPLEMENT_ITEMS> transfer;
};

