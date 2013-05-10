#pragma once

#include <clever/clever.hpp>

class GridConfig {

public:

	GridConfig(uint _nLayers, uint _nSectorsZ, uint _nSectorsPhi) :
		nLayers(_nLayers), nSectorsZ(_nSectorsZ), nSectorsPhi(_nSectorsPhi),
		m_boundaryValuesZ(NULL), m_boundaryValuesPhi(NULL){

		sectorSizeZ = (GridConfig::MAX_Z - GridConfig::MIN_Z) / this->nSectorsZ;
		for(uint i = 0; i <= this->nSectorsZ; ++i){
			boundaryValuesZ.push_back(GridConfig::MIN_Z + i*sectorSizeZ);
		}

		sectorSizePhi = (GridConfig::MAX_PHI - GridConfig::MIN_PHI) / this->nSectorsPhi;
		for(uint i = 0; i <= this->nSectorsPhi; ++i){
			boundaryValuesPhi.push_back(GridConfig::MIN_PHI + i*sectorSizePhi);
		}
	}

	~GridConfig() {
		delete m_boundaryValuesZ;
		delete m_boundaryValuesPhi;
	}

	void upload(clever::icontext & ctx){
		m_boundaryValuesZ = new clever::vector<cl_float, 1>(boundaryValuesZ, ctx);
		m_boundaryValuesPhi = new clever::vector<cl_float, 1>(boundaryValuesPhi, ctx);
	}

	cl_mem getBoundaryValuesZ() const {
		if(m_boundaryValuesZ != NULL)
			return m_boundaryValuesZ->get_mem();

		return NULL;
	}

	cl_mem getBoundaryValuesPhi() const {
		if(m_boundaryValuesPhi != NULL)
			return m_boundaryValuesPhi->get_mem();

		return NULL;
	}

public:
	uint nLayers;
	uint nSectorsZ;
	float sectorSizeZ;
	uint nSectorsPhi;
	float sectorSizePhi;

	std::vector<cl_float> boundaryValuesZ;
	std::vector<cl_float> boundaryValuesPhi;

private:
	clever::vector<cl_float, 1> * m_boundaryValuesZ;
	clever::vector<cl_float, 1> * m_boundaryValuesPhi;

public:

	static constexpr float MIN_Z = -50; //-300;
	static constexpr float MAX_Z = 50; //300;

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

	Grid(uint _nLayers, uint _nSectorsZ, uint _nSectorsPhi, uint _nEvents = 1) :
			clever::Collection<GRID_ITEMS>(_nEvents *_nLayers*(_nSectorsZ+1)*(_nSectorsPhi+1)),
			config(_nLayers, _nSectorsZ, _nSectorsPhi)
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
	LayerGrid(const Grid & grid, uint layer, uint maxLayers, uint evt)
	: nSectorsZ(grid.config.nSectorsZ), nSectorsPhi(grid.config.nSectorsPhi)
	{

		uint entriesPerLayer = (nSectorsZ+1)*(nSectorsPhi+1);
		uint entriesPerEvent = maxLayers*entriesPerLayer;

		for(uint i = 0; i <= nSectorsZ; ++i){
			for(uint j = 0; j <= nSectorsPhi; ++j){
				GridEntry entry(grid, evt * entriesPerEvent + (layer-1)*entriesPerLayer+i*(nSectorsPhi+1)+j);
				m_data.push_back(entry.idx());
			}
		}

	}

	uint operator()(uint z, uint p){
		return m_data[z*(nSectorsPhi+1)+p];
	}

	uint operator()(uint z){
			return m_data[z*(nSectorsPhi+1)];
	}

public:
	uint nSectorsZ;
	uint nSectorsPhi;

private:
	std::vector<uint> m_data;

};
