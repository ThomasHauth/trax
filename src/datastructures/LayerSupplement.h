#pragma once

#include <vector>

class LayerInformation {

public:

	LayerInformation(uint nSectors)
		: nHits(0), offset(0), sectorBorders(nSectors)
	{ }

public:
	uint nHits;
	uint offset;
	std::vector<uint> sectorBorders;

};

class LayerSupplement : public std::vector<LayerInformation>{

public:

	LayerSupplement(uint nLayers, uint nSectors){

		for(uint i = 0; i < nLayers; ++i)
			this->push_back(LayerInformation(nSectors));
	}

	LayerInformation & operator[](uint i){
		return this->at(i);
	}

	const LayerInformation & operator[](uint i) const {
			return this->at(i);
		}

};
