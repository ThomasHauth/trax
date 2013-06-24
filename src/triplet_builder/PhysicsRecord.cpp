/*
 * PhysicsRecord.h
 *
 *  Created on: May 23, 2013
 *      Author: dfunke
 */

#include "PhysicsRecord.h"

void tBinnedData::operator+=(const tBinnedData c){
	valid += c.valid;
	clones += c.clones;
	fake += c.fake;
	missed += c.missed;

	n += c.n;

	long double delta = c.efficiencyMean - efficiencyMean;
	efficiencyMean += (c.valid + c.missed)*delta / Utils::clamp((valid + missed)); //mc truth defines n here
	efficiencyVar +=  (c.valid + c.missed)*delta*(c.efficiencyMean - efficiencyMean) + c.efficiencyVar;

	delta = c.fakeRateMean - fakeRateMean;
	fakeRateMean += c.n*delta / Utils::clamp(n);
	fakeRateVar +=  c.n*delta*(c.fakeRateMean - fakeRateMean) + c.fakeRateVar;

	delta = c.cloneRateMean - cloneRateMean;
	cloneRateMean += c.n*delta / Utils::clamp(n);
	cloneRateVar +=  c.n*delta*(c.cloneRateMean - cloneRateMean) + c.cloneRateVar;
}

void tBinnedData::fill(){

	efficiencyMean = ((double) valid) / Utils::clamp(valid + missed);
	fakeRateMean = ((double) fake / Utils::clamp(valid + fake + clones + misc));
	cloneRateMean = ((double) clones / Utils::clamp(valid + fake + clones + misc));

	n = valid + fake + clones + misc;

}

tCircleParams PhysicsRecord::getCircleParams(const Hit & p1, const Hit & p2, const Hit & p3) const{

	tCircleParams params;

	//circle fit
	//map points to parabloid: (x,y) -> (x,y,x^2+y^2)
	fVector3 pP1 (p1.globalX(),
			p1.globalY(),
			p1.globalX() * p1.globalX() + p1.globalY() * p1.globalY());

	fVector3 pP2 (p2.globalX(),
			p2.globalY(),
			p2.globalX() * p2.globalX() + p2.globalY() * p2.globalY());

	fVector3 pP3 (p3.globalX(),
			p3.globalY(),
			p3.globalX() * p3.globalX() + p3.globalY() * p3.globalY());

	//span two vectors
	fVector3 a(pP2.x - pP1.x, pP2.y - pP1.y, pP2.z - pP1.z);
	fVector3 b(pP3.x - pP1.x, pP3.y - pP1.y, pP3.z - pP1.z);

	//compute unit cross product
	fVector3 n(a.y*b.z - a.z*b.y,
			a.z*b.x - a.x*b.z,
			a.x*b.y - a.y*b.x );
	double value = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
	n.x /= value; n.y /= value; n.z /= value;

	//formula for orign and radius of circle from Strandlie et al.
	params.center = fVector2((-n.x) / (2*n.z),
			(-n.y) / (2*n.z));

	double c = -(n.x*pP1.x + n.y*pP1.y + n.z*pP1.z);

	params.radius = sqrt((1 - n.z*n.z - 4 * c * n.z) / (4*n.z*n.z));

	return params;
}

void PhysicsRecord::fillData(const TrackletCollection& tracklets,
		const HitCollection::tTrackList& mcTruth, const HitCollection& hits, uint nLayerTriplets) {

	LOG << "Evaluating event " << event << " layer triplet " << layerTriplet  << std::endl;

	//std::ofstream histo("ptCalc", std::ios::app);

	//uint cPt = 0; uint swPt = 0; uint rwPt = 0;

	std::set<uint> foundTracks;
	uint fakeTracks = 0;
	uint foundClones = 0;

	uint nFoundTracklets = tracklets.getTrackletOffsets()[event * nLayerTriplets + layerTriplet + 1] - tracklets.getTrackletOffsets()[event * nLayerTriplets + layerTriplet];

	LOG << "Found " << nFoundTracklets << " triplets:" << std::endl;
	for(uint i = tracklets.getTrackletOffsets()[event * nLayerTriplets + layerTriplet]; i < tracklets.getTrackletOffsets()[event * nLayerTriplets + layerTriplet + 1]; ++i){
		Tracklet tracklet(tracklets, i);

		float tEta = getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3()));
		float tPt = getPt(Hit(hits,tracklet.hit1()),  Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));

		if(tracklet.isValid(hits)){
			//valid triplet

			if(mcTruth.find(tracklet.trackId(hits)) != mcTruth.end()){ //  ensure that it is a "findable track" otherwise efficiency greater one possible

				bool inserted = foundTracks.insert(tracklet.trackId(hits)).second;

				if(inserted){ //found for the first time
					eta[getEtaBin(tEta)].valid++;
					pt[getPtBin(tPt)].valid++;

					/*double calculatedPt = getPt(Hit(hits,tracklet.hit1()),  Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
				double mcPt = mcTruth.at(tracklet.trackId(hits))[0].simtrackpt();

				histo << fabs(calculatedPt - mcPt) / mcPt << std::endl;

				if((fabs(calculatedPt - mcPt) / mcPt) < 0.1)
					cPt++;
				else if((fabs(calculatedPt - mcPt) / mcPt) < 0.9)
					swPt++;
				else
					rwPt++;*/

					VLOG << zkr::cc::fore::green;
					VLOG << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
					VLOG << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
					VLOG << zkr::cc::console << std::endl;
				} else {
					//clone
					++foundClones;

					eta[getEtaBin(tEta)].clones++;
					pt[getPtBin(tPt)].clones++;

					/*double calculatedPt = getPt(Hit(hits,tracklet.hit1()),  Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
				double mcPt = mcTruth.at(tracklet.trackId(hits))[0].simtrackpt();

				histo << fabs(calculatedPt - mcPt) / mcPt << std::endl;

				if((fabs(calculatedPt - mcPt) / mcPt) < 0.1)
					cPt++;
				else if((fabs(calculatedPt - mcPt) / mcPt) < 0.9)
					swPt++;
				else
					rwPt++;*/

					VLOG << zkr::cc::fore::yellow;
					VLOG << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
					VLOG << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
					VLOG << zkr::cc::console << std::endl;
				}
			} else { //not in findable tracks definition but still valid
				eta[getEtaBin(tEta)].misc++;
				pt[getPtBin(tPt)].misc++;
			}
		}
		else {
			//fake triplet
			++fakeTracks;

			eta[getEtaBin(tEta)].fake++;
			pt[getPtBin(tPt)].fake++;

			VLOG << zkr::cc::fore::red;
			VLOG << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
			VLOG << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
			VLOG << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
			VLOG << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
			VLOG << zkr::cc::console << std::endl;

		}
	}

	//std::cout << "\tINTERMEIDATE: correctly calculated: " << cPt << " slightly wrongly calculated: " << swPt << " really wrong " << rwPt << std::endl;

	//output not found tracks
	for(auto vTrack : mcTruth) {
		if( foundTracks.find(vTrack.first) == foundTracks.end()){
			VLOG << "Didn't find track " << vTrack.first << std::endl;

			PHitWrapper innerHit(vTrack.second[0]);
			PHitWrapper middleHit(vTrack.second[floor(vTrack.second.size()/2)]);
			PHitWrapper outerHit(vTrack.second[vTrack.second.size()-1]);

			/*double calculatedPt = getPt(innerHit,  middleHit, outerHit);
			double mcPt = vTrack.second[0].simtrackpt();

			histo << fabs(calculatedPt - mcPt) / mcPt << std::endl;

			if((fabs(calculatedPt - mcPt) / mcPt) < 0.1)
				cPt++;
			else if((fabs(calculatedPt - mcPt) / mcPt) < 0.9)
				swPt++;
			else
				rwPt++;*/

			eta[getEtaBin(getEta(innerHit, outerHit))].missed++;
			pt[getPtBin(getPt(innerHit,  middleHit, outerHit))].missed++;
		}
	}


	efficiencyMean = ((double) foundTracks.size()) / Utils::clamp(mcTruth.size());
	fakeRateMean = ((double) fakeTracks) / Utils::clamp(nFoundTracklets);
	cloneRateMean = ((double) foundClones) / Utils::clamp(nFoundTracklets);

	eta.fill();
	pt.fill();

	//std::cout << "correctly calculated: " << cPt << " slightly wrongly calculated: " << swPt << " really wrong " << rwPt << std::endl;

	LOG << "Efficiency: " << efficiencyMean  << " FakeRate: " << fakeRateMean << " CloneRate: " << cloneRateMean << std::endl;

	//histo.close();

}

void mergeHistograms(tHistogram & t1, const tHistogram & t2){
	for(auto t : t2){
		if(t1.binWidth == t2.binWidth)
			t1[t.first] += t.second;
		else {
			double bin = t.first * t2.binWidth + t2.binWidth / 2; //use bin center;
			bin = t1.binWidth * floor(bin / t1.binWidth);
			t1[bin] += t.second;
		}
	}
}

void PhysicsRecord::merge(const PhysicsRecord& c) {

	mergeHistograms(eta, c.eta);
	mergeHistograms(pt, c.pt);

	n += c.n;

	long double delta = c.efficiencyMean - efficiencyMean;
	efficiencyMean += c.n*delta / Utils::clamp(n);
	efficiencyVar +=  c.n*delta*(c.efficiencyMean - efficiencyMean) + c.efficiencyVar;

	delta = c.fakeRateMean - fakeRateMean;
	fakeRateMean += c.n*delta / Utils::clamp(n);
	fakeRateVar +=  c.n*delta*(c.fakeRateMean - fakeRateMean) + c.fakeRateVar;

	delta = c.cloneRateMean - cloneRateMean;
	cloneRateMean += c.n*delta / Utils::clamp(n);
	cloneRateVar +=  c.n*delta*(c.cloneRateMean - cloneRateMean) + c.cloneRateVar;

}

std::string PhysicsRecord::csvDump(std::string outputDir) const {

	std::stringstream s;

	s << Utils::csv({layerTriplet, n}) << SEP; //header
	s << Utils::csv({efficiencyMean, toVar(efficiencyVar)}) << SEP; //efficiency
	s << Utils::csv({fakeRateMean, toVar(fakeRateVar)}) << SEP; //fake rate
	s << Utils::csv({cloneRateMean, toVar(cloneRateVar)}); //clone rate

	if(outputDir != ""){ //dump histograms
		//eta
		std::stringstream outputFileEta;
		outputFileEta << outputDir << "/eta_" << layerTriplet << ".csv";

		std::ofstream etaFile(outputFileEta.str(), std::ios::trunc);
		etaFile << eta.csvDump();
		etaFile.close();

		//pt
		std::stringstream outputFilePt;
		outputFilePt << outputDir << "/pt_" << layerTriplet << ".csv";

		std::ofstream ptFile(outputFilePt.str(), std::ios::trunc);
		ptFile << pt.csvDump();
		ptFile.close();
	}

	return s.str();

}

double PhysicsRecord::getTIP(const Hit & p1, const Hit & p2, const Hit & p3) const {

	tCircleParams params = getCircleParams(p1, p2, p3);

	//find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
	fVector2 v(-params.center.x, -params.center.y);
	double value = sqrt(v.x*v.x+v.y*v.y);
	v.x /= value; v.y /= value;;

	fVector2 pCA(params.center.x + params.radius*v.x,
			params.center.y + params.radius*v.y);

	//TIP = distance of point of closest approach to origin
	double tip = sqrt(pCA.x*pCA.x + pCA.y*pCA.y);

	return tip;
}

void PhysicsRecords::addRecord(const PhysicsRecord& r) {

	for(uint i = 0; i < records.size(); ++i){

		if(records[i] == r){
			records[i].merge(r);
			return;
		}
	}

	records.push_back(r);

}

void PhysicsRecords::merge(const PhysicsRecords& c) {
	for(uint i = 0; i < c.records.size(); ++i){
		addRecord(c.records[i]);
	}
}

std::string PhysicsRecords::csvDump(std::string outputDir) const {

	std::stringstream s;

	//header
	s << Utils::csv({"layerTriplet", "n"}) << SEP; //header
	s << Utils::csv({"eff", "effVar"}) << SEP; //efficiency
	s << Utils::csv({"fr", "frVar"}) << SEP; //fake rate
	s << Utils::csv({"cr", "crVar", }); //clone rate
	s << std::endl;

	PhysicsRecord total(records[0]);
	total.layerTriplet = records.size()+1;
	for(uint i = 0; i < records.size(); ++i){
		s << records[i].csvDump(outputDir) << std::endl;

		if(i > 0)
			total.merge(records[i]);
	}

	s << total.csvDump(outputDir) << std::endl;

	return s.str();

}

std::string tBinnedData::csvDump() const {
	std::stringstream s;

	s << Utils::csv({valid, fake, clones, missed, misc}) << SEP << Utils::csv({efficiencyMean, toVar(efficiencyVar), fakeRateMean, toVar(fakeRateVar), cloneRateMean, toVar(cloneRateVar)});

	return s.str();
}

std::string tHistogram::csvDump() const {

	std::stringstream s;

	s << Utils::csv({"bin", "valid", "fake", "clones", "missed", "misc", "efficiencyMean", "efficiencyVar", "fakeRateMean", "fakeRateVar", "cloneRateMean", "cloneRateVar"}) << std::endl;

	for(auto it = begin(); it != end(); ++it){
		s << it->first << SEP << it->second.csvDump() << std::endl;
	}

	return s.str();

}
