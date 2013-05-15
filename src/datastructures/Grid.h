#pragma once

#include <clever/clever.hpp>

class GridConfig {

public:

	float sectorSizeZ() const {
		return (GridConfig::MAX_Z - GridConfig::MIN_Z) / this->nSectorsZ;
	}

	float sectorSizePhi() const {
		return (GridConfig::MAX_PHI - GridConfig::MIN_PHI) / this->nSectorsPhi;
	}

	float boundaryValueZ(uint z) const {
		return GridConfig::MIN_Z + z*sectorSizeZ();
	}

	float boundaryValuePhi(uint p) const {
			return GridConfig::MIN_PHI + p*sectorSizePhi();
		}

public:
	uint nEvents;
	uint nLayers;
	uint nSectorsZ;
	uint nSectorsPhi;

public:

	static constexpr float MIN_Z = -300; //for PXB -50
	static constexpr float MAX_Z = 300; //for PXB 50;

	static constexpr float MIN_PHI = -M_PI;
	static constexpr float MAX_PHI =  M_PI;

};

struct Boundary: public clever::UIntItem
{
};

#define GRID_ITEMS Boundary

typedef clever::Collection<GRID_ITEMS> GridItems;

class Grid: public GridItems
{
public:
	typedef GridItems dataitems_type;

	Grid(const GridConfig & cfg) :
			clever::Collection<GRID_ITEMS>(cfg.nEvents *cfg.nLayers*(cfg.nSectorsZ+1)*(cfg.nSectorsPhi+1)),
			config(cfg)
	{
		size_t items = size();
		for(uint i = 0; i < items; ++i){
			addWithValue(0);
		}
	}

public:
	clever::OpenCLTransfer<GRID_ITEMS> transfer;
	GridConfig config;

};

class GridEntry: private clever::CollectionView<Grid>
{
public:
// get a pointer to one hit in the collection
	GridEntry(const Grid & collection, index_type i) :
			clever::CollectionView<Grid>(collection, i)
	{

	}

// create a new hit in the collection
	GridEntry(Grid & collection) :
			clever::CollectionView<Grid>(collection)
	{
	}

	float idx() const
	{
		return getValue<Boundary>();
	}

};

class LayerGrid
{
public:
	LayerGrid(const Grid & grid, uint layer, uint evt)
	: nSectorsZ(grid.config.nSectorsZ), nSectorsPhi(grid.config.nSectorsPhi)
	{

		uint entriesPerLayer = (nSectorsZ+1)*(nSectorsPhi+1);
		uint entriesPerEvent = grid.config.nLayers*entriesPerLayer;

		for(uint i = 0; i <= nSectorsZ; ++i){
			for(uint j = 0; j <= nSectorsPhi; ++j){
				GridEntry entry(grid, evt * entriesPerEvent + (layer-1)*entriesPerLayer+i*(nSectorsPhi+1)+j);
				m_data.push_back(entry.idx());
			}
		}

	}

	uint operator()(uint z, uint p) const {
		return m_data[z*(nSectorsPhi+1)+p];
	}

	uint operator()(uint z) const {
			return m_data[z*(nSectorsPhi+1)];
	}

	uint size() const {
		return (*this)(nSectorsZ, nSectorsPhi) - (*this)(0,0);
	}

public:
	uint nSectorsZ;
	uint nSectorsPhi;

private:
	std::vector<uint> m_data;

};
