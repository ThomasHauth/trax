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

struct sigmaZ : public clever::FloatItem { };

struct dPhiCut : public clever::FloatItem { };

struct sigmaPhi : public clever::FloatItem { };

struct tipCut : public clever::FloatItem { };

struct pairSpreadZ : public clever::UIntItem { };

struct z0 : public clever::FloatItem { };

struct pairSpreadPhi : public clever::UIntItem { };

struct nCandidates : public clever::UIntItem { };

#define TRIPLET_CONFIGURATION_COLLECTION_ITEMS Layer1, Layer2, Layer3, dThetaCut, sigmaZ, dPhiCut, sigmaPhi, tipCut, pairSpreadZ, z0, pairSpreadPhi, nCandidates

typedef clever::Collection<TRIPLET_CONFIGURATION_COLLECTION_ITEMS> TripletConfigurationItems;

class TripletConfiguration;

class TripletConfigurations: public TripletConfigurationItems
{
public:
	typedef TripletConfigurations dataitems_type;

	TripletConfigurations(float minPt) : minPt_(minPt) {}

	TripletConfigurations(int items, float minPt) :
		clever::Collection<TRIPLET_CONFIGURATION_COLLECTION_ITEMS>(items),
		minPt_(minPt) {	}

	TripletConfiguration operator[](uint i);
	const TripletConfiguration operator[](uint i) const;

	uint calculatePairSpreadZ(uint layer1, uint layer2, const Grid & grid, const GeometrySupplement & geom) const;
	uint calculatePairSpreadPhi(uint layer1, uint layer2, float minPt, float d0, float Bz, const Grid & grid, const GeometrySupplement & geom) const;

	float minRadiusCurvature() const {
		return minRadiusCurvature(minPt_);
	}

	float minRadiusCurvature(float minPt) const {
		return minRadiusCurvature(minPt, IDEAL_BZ);
	}

	float minRadiusCurvature(float minPt, float Bz) const ;

	uint loadTripletConfigurationFromFile(std::string filename, int n = -1);

public:
	clever::OpenCLTransfer<TRIPLET_CONFIGURATION_COLLECTION_ITEMS> transfer;
	const double IDEAL_BZ = 3.8112;
	float minPt_;
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
