#pragma once

#include <vector>
#include <clever/clever.hpp>
#include "CommonTypes.h"

struct Layer1: public clever::UIntItem
{
};

struct Layer2: public clever::UIntItem
{
};

struct Layer3: public clever::UIntItem
{
};

#define LAYER_TRIPLETS_COLLECTION_ITEMS Layer1, Layer2, Layer3

typedef clever::Collection<LAYER_TRIPLETS_COLLECTION_ITEMS> LayerTripletsItems;

class LayerTriplet;

class LayerTriplets: public LayerTripletsItems
{
public:
	typedef LayerTriplets dataitems_type;

	LayerTriplet operator[](uint i);
	const LayerTriplet operator[](uint i) const;

public:
	clever::OpenCLTransfer<LAYER_TRIPLETS_COLLECTION_ITEMS> transfer;
};

class LayerTriplet: private clever::CollectionView<LayerTriplets>
{
public:
// get a pointer to one hit in the collection
	LayerTriplet(LayerTriplets & collection, index_type i) :
			clever::CollectionView<LayerTriplets>(collection, i)
	{
	}

	LayerTriplet(const LayerTriplets & collection, index_type i) :
		clever::CollectionView<LayerTriplets>(collection, i)
		{
		}

// create a new hit in the collection
	LayerTriplet(LayerTriplets & collection) :
			clever::CollectionView<LayerTriplets>(collection)
	{
	}

	uint layer1() const
	{
		return getValue<Layer1>();
	}

	uint layer2() const
	{
		return getValue<Layer2>();
	}

	uint layer3() const
	{
		return getValue<Layer3>();
	}


};
