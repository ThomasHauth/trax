#pragma once

#include <vector>
#include <clever/clever.hpp>
#include "CommonTypes.h"
#include <datastructures/Grid.h>
#include <datastructures/GeometrySupplement.h>

struct Layer1: public clever::UIntItem {};

struct Layer2: public clever::UIntItem {};

struct Layer3: public clever::UIntItem {};

struct dThetaCut: public clever::FloatItem { };

struct dThetaWindow : public clever::FloatItem { };

struct dPhiCut : public clever::FloatItem { };

struct dPhiWindow : public clever::FloatItem { };

struct tipCut : public clever::FloatItem { };

struct pairSpreadZ : public clever::UIntItem { };

struct pairSpreadPhi : public clever::UIntItem { };

#define TRIPLET_CONFIGURATION_COLLECTION_ITEMS Layer1, Layer2, Layer3, dThetaCut, dThetaWindow, dPhiCut, dPhiWindow, tipCut, pairSpreadZ, pairSpreadPhi

typedef clever::Collection<TRIPLET_CONFIGURATION_COLLECTION_ITEMS> TripletConfigurationItems;

class TripletConfiguration;

class TripletConfigurations: public TripletConfigurationItems
{
public:
	typedef TripletConfigurations dataitems_type;

	TripletConfigurations() {}

	TripletConfigurations(int items) :
		clever::Collection<TRIPLET_CONFIGURATION_COLLECTION_ITEMS>(items) {	}

	TripletConfiguration operator[](uint i);
	const TripletConfiguration operator[](uint i) const;

	uint calculatePairSpreadZ(uint layer1, uint layer2, const Grid & grid, const GeometrySupplement & geom);
	uint calculatePairSpreadPhi(uint layer1, uint layer2, float minPt, float d0, float Bz, const Grid & grid, const GeometrySupplement & geom);

	uint loadTripletConfigurationFromFile(std::string filename, int n = -1);

public:
	clever::OpenCLTransfer<TRIPLET_CONFIGURATION_COLLECTION_ITEMS> transfer;
};

class TripletConfiguration: public clever::CollectionView<TripletConfigurations>
{
public:
// get a pointer to one hit in the collection
	TripletConfiguration(TripletConfigurations & collection, index_type i) :
			clever::CollectionView<TripletConfigurations>(collection, i)
	{
	}

	TripletConfiguration(const TripletConfigurations & collection, index_type i) :
		clever::CollectionView<TripletConfigurations>(collection, i)
		{
		}

// create a new hit in the collection
	TripletConfiguration(TripletConfigurations & collection) :
			clever::CollectionView<TripletConfigurations>(collection)
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
