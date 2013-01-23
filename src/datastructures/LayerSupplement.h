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

private:
	mutable std::vector<uint> layerHits;
	mutable std::vector<uint> layerOffsets;

};
