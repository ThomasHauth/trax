#pragma once

#include <clever/clever.hpp>

struct Boundary: public clever::UIntItem
{
};

#define GRID_ITEMS Boundary

typedef clever::Collection<GRID_ITEMS> GridItems;

class Grid: public GridItems
{
public:
	typedef GridItems dataitems_type;

	Grid(uint _nLayers, uint _nSectorsZ, uint _nSectorsPhi) :
			clever::Collection<GRID_ITEMS>(_nLayers*(_nSectorsZ+1)*(_nSectorsPhi+1)),
			nLayers(_nLayers), nSectorsZ(_nSectorsZ), nSectorsPhi(_nSectorsPhi)
	{

	}

public:
	uint nLayers;
	uint nSectorsZ;
	uint nSectorsPhi;

	clever::OpenCLTransfer<GRID_ITEMS> transfer;

	static constexpr float MIN_Z = -300;
	static constexpr float MAX_Z = 300;

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
	LayerGrid(const Grid & grid, uint layer)
	: nSectorsZ(grid.nSectorsZ), nSectorsPhi(grid.nSectorsPhi)
	{

		uint entriesPerLayer = (nSectorsZ+1)*(nSectorsPhi+1);

		for(uint i = 0; i <= nSectorsZ; ++i){
			for(uint j = 0; j <= nSectorsPhi; ++j){
				GridEntry entry(grid, layer*entriesPerLayer+i*(nSectorsPhi+1)+j);
				m_data.push_back(entry.idx());
			}
		}

	}

	uint operator()(uint i, uint j){
		return m_data[i*(nSectorsPhi+1)+j];
	}

	uint operator()(uint i){
			return m_data[i*(nSectorsPhi+1)];
	}

public:
	uint nSectorsZ;
	uint nSectorsPhi;

private:
	std::vector<uint> m_data;

};
