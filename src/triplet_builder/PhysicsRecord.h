/*
 * PhysicsRecord.h
 *
 *  Created on: May 23, 2013
 *      Author: dfunke
 */

#pragma once

#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <fcntl.h>

#include <datastructures/HitCollection.h>
#include <datastructures/serialize/Event.pb.h>
#include <datastructures/TrackletCollection.h>
#include "lib/ccolor.h"

#include "utils.h"

struct fVector2 {
	double x;
	double y;

	fVector2(double _x = 0, double _y = 0) {
		x = _x;
		y = _y;
	}
};

struct fVector3 {
	double x;
	double y;
	double z;

	fVector3(double _x = 0, double _y = 0, double _z = 0) {
		x = _x;
		y = _y;
		z = _z;
	}
};

struct tBinnedData{
	uint valid;
	uint clones;
	uint fake;
	uint missed;

	long double efficiencyMean; long double efficiencyVar;
	long double fakeRateMean; long double fakeRateVar;
	long double cloneRateMean; long double cloneRateVar;
	uint n;

	tBinnedData() : valid(0), clones(0), fake(0), missed(0),
			efficiencyMean(0), efficiencyVar(0), fakeRateMean(0), fakeRateVar(0),
			cloneRateMean(0), cloneRateVar(0), n(1){}

	void fill();

	long double toVar(long double x) const{
		return n > 1 ? std::sqrt(x / Utils::clamp(n - 1)) : 0;
	}

	void operator+=(const tBinnedData t);
	std::string csvDump() const;
};

class tHistogram : public std::map<double, tBinnedData> {

public:
	tHistogram(double binWidth_) : binWidth(binWidth_) { }

	std::string csvDump() const;

	void fill() {
		for(auto & t : *this)
			t.second.fill();
	}

public:
	double binWidth;

};

struct tCircleParams{
	fVector2 center;
	double radius;
};

class PhysicsRecord{

public:
	PhysicsRecord(uint e, uint lt) :
		event(e), layerTriplet(lt), efficiencyMean(0), efficiencyVar(0), fakeRateMean(0), fakeRateVar(0),
		cloneRateMean(0), cloneRateVar(0), n(1),
		pt(0.1), eta(0.1) {	}

	void fillData(const TrackletCollection & tracklets, const HitCollection::tTrackList & mcTruth, const HitCollection & hits, uint nLayerTriplets);

	void merge(const PhysicsRecord & c);

	long double toVar(long double x) const{
		return n > 1 ? std::sqrt(x / Utils::clamp(n - 1)) : 0;
	}

	std::string csvDump(std::string outputDir = "") const;

public:

	uint event;

	uint layerTriplet;

	long double efficiencyMean; long double efficiencyVar;
	long double fakeRateMean; long double fakeRateVar;
	long double cloneRateMean; long double cloneRateVar;
	uint n;

	tHistogram pt;
	tHistogram eta;

	bool operator==(const PhysicsRecord & a){
		return layerTriplet == a.layerTriplet;
	}


public:
	// Ideal Magnetic Field [T] (-0,-0, 3.8112)
	const double BZ = 3.8112;
	// e = 1.602177×10^-19 C  (coulombs)
	const double Q = 1.602177E-19;
	// c = 2.998×10^8 m/s  (meters per second)
	const double C = 2.998E8;
	// 1 GeV/c = 5.344286×10^-19 J s/m  (joule seconds per meter)
	const double GEV_C = 5.344286E-19;

private:

	tCircleParams getCircleParams(const Hit & p1, const Hit & p2, const Hit & p3) const;

	double getTIP(const Hit & p1, const Hit & p2, const Hit & p3) const;

	double getEta(const Hit & innerHit, const Hit & outerHit) const {
		fVector3 p (outerHit.globalX() - innerHit.globalX(), outerHit.globalY() - innerHit.globalY(), outerHit.globalZ() - innerHit.globalZ());
		return getEta(p);
	}

	double getEta(fVector3 p) const {
		double t(p.z/std::sqrt(p.x*p.x+p.y*p.y));
		return asinh(t);
	}

	double getEtaBin(double eta_) const {
		return eta.binWidth*floor(eta_/eta.binWidth);
	}

	double getPt(const Hit & p1, const Hit & p2, const Hit & p3) const {
		return getPt(getCircleParams(p1, p2, p3));
	}

	double getPt(tCircleParams params) const { //in [GeV/c]
		return Q * BZ * (params.radius*1E-2) / GEV_C; //convert radius from cm to meter and then to GEV/c
	}

	double getPtBin(double pt_) const {
		return pt.binWidth * floor(pt_/pt.binWidth);
	}

};

class PhysicsRecords {

private:
	std::vector<PhysicsRecord> records;

public:
	void addRecord(const PhysicsRecord & r);
	void merge(const PhysicsRecords & c);

	std::string csvDump(std::string outputDir = "") const;

};
