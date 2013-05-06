#pragma once

#include <vector>
#include <clever/clever.hpp>
#include "CommonTypes.h"


#define EVENT_SUPPLEMENT_COLLECTION_ITEMS NHits, Offset

typedef clever::Collection<EVENT_SUPPLEMENT_COLLECTION_ITEMS> EventSupplementItems;

class EventInfo;

class EventSupplement: public EventSupplementItems
{
public:
	typedef EventSupplement dataitems_type;

	EventSupplement(uint _nEvents = 1) :
			clever::Collection<EVENT_SUPPLEMENT_COLLECTION_ITEMS>(_nEvents), nEvents(_nEvents)
	{

	}

	EventInfo operator[](uint i);
	const EventInfo operator[](uint i) const;

public:
	uint nEvents;

	clever::OpenCLTransfer<EVENT_SUPPLEMENT_COLLECTION_ITEMS> transfer;
};

class EventInfo: private clever::CollectionView<EventSupplement>
{
public:
// get a pointer to one hit in the collection
	EventInfo(EventSupplement & collection, index_type i) :
			clever::CollectionView<EventSupplement>(collection, i)
	{
	}

	EventInfo(const EventSupplement & collection, index_type i) :
		clever::CollectionView<EventSupplement>(collection, i)
		{
		}

// create a new hit in the collection
	EventInfo(EventSupplement & collection) :
			clever::CollectionView<EventSupplement>(collection)
	{
	}

	uint getNHits() const
	{
		return getValue<NHits>();
	}

	uint getOffset() const
	{
		return getValue<Offset>();
	}

	void setNHits(uint i){
		setValue<NHits>(i);
	}

	void setOffset(uint i){
		setValue<Offset>(i);
	}

};

/*class LayerInformation {

public:

	LayerInformation(uint nSectorsZ, uint nSectorsPhi)
		: nHits(0), offset(0)
	{

		sectorBorders = new uint[nSectorsZ*nSectorsPhi];

	}

public:
	uint nHits;
	uint offset;
	uint *sectorBorders;

};

class LayerSupplement : public std::vector<LayerInformation>{

public:

	LayerSupplement(uint _nLayers, uint _nSectorsZ, uint _nSectorsPhi)
	: nLayers(_nLayers), nSectorsZ(_nSectorsZ), nSectorsPhi(_nSectorsPhi) {

		for(uint i = 0; i < nLayers; ++i)
			this->push_back(LayerInformation(nSectorsZ, nSectorsPhi));
	}

	LayerInformation & operator[](uint i){
		//invalidate

		return this->at(i);
	}

	const LayerInformation & operator[](uint i) const {
			return this->at(i);
	}

	std::vector<uint> getLayerHits() const {

		if(layerHits.size() != this->size()){
			layerHits.clear();
			for(const LayerInformation & info : *this)
				layerHits.push_back(info.nHits);
		}

		return layerHits;

	}

	std::vector<uint> getLayerOffsets() const {

		if(layerOffsets.size() != this->size()){
			layerOffsets.clear();
			for(const LayerInformation & info : *this)
				layerOffsets.push_back(info.offset);
		}

		return layerOffsets;

	}

public:
	uint nLayers;
	uint nSectorsZ;
	uint nSectorsPhi;

	static constexpr float MIN_Z = -300;
	static constexpr float MAX_Z = 300;

private:
	mutable std::vector<uint> layerHits;
	mutable std::vector<uint> layerOffsets;

};*/
