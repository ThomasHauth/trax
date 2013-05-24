/*
 * LayerSupplement.cpp
 *
 *  Created on: Jan 30, 2013
 *      Author: dfunke
 */

#include "TripletConfiguration.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

TripletConfiguration TripletConfigurations::operator[](uint i){
	return TripletConfiguration(*this, i);
}

const TripletConfiguration TripletConfigurations::operator[](uint i) const {
	return TripletConfiguration(*this, i);
}

uint TripletConfigurations::calculatePairSpreadZ(uint layer1, uint layer2, const Grid & grid, const GeometrySupplement & geom){

//	  \abs{\Delta_z} \leq (r_2-r_1) \cot\(\num{0.05}\pi) for \abs{eta} < 2.5

	LayerGeometry gLayer1(geom, layer1);
	LayerGeometry gLayer2(geom, layer2);

	float dZ = (gLayer2.maxRadius() - gLayer1.minRadius())*std::tan(M_PI_2 - 0.05 * M_PI);

	//calculate number of grid cells covered by dZ
	return ceil(dZ / grid.config.sectorSizeZ());

}


uint TripletConfigurations::calculatePairSpreadPhi(uint layer1, uint layer2, float minPt, float d0, float Bz, const Grid & grid, const GeometrySupplement & geom){

//	minimum radius \frac{1}{r_\mathrm{min}} = \kappa_\mathrm{max} = \frac{q B_z}{p_{T, \mathrm{min}}}.
//	maximum dPhi
//	due to curvature \abs{\Delta_\phi} \leq  \abs{\arccos\left( \frac{r_2}{2 r_\mathrm{min}}\right)} - \arccos\left(\frac{r_1}{2 r_\mathrm{min}}\right)}
//	due to TIP       + \arctan\left(\frac{d_0 (r_2 - r_1)}{r_1 r_2}\right)}

		// e = 1.602177×10^-19 C  (coulombs)
		const double Q = 1.602177E-19;
		// 1 GeV/c = 5.344286×10^-19 J s/m  (joule seconds per meter)
		const double GEV_C = 5.344286E-19;

	float rMin = minPt * GEV_C / (Bz * Q);

	LayerGeometry gLayer1(geom, layer1);
	LayerGeometry gLayer2(geom, layer2);

	float dPhi = std::fabs(std::acos(gLayer2.maxRadius() / (2 * rMin)) - std::acos(gLayer1.minRadius() / (2 * rMin)));
	dPhi += std::atan(d0 * (gLayer2.maxRadius() - gLayer1.minRadius()))/(gLayer2.maxRadius() * gLayer2.minRadius());

	//calcualte number of grid cells covered by dPhi
	return ceil(dPhi / grid.config.sectorSizePhi());

}

uint TripletConfigurations::loadTripletConfigurationFromFile(std::string filename, int n){

	using boost::property_tree::ptree;
	ptree pt;

	read_xml(filename, pt);

	int added = 0;
	uint maxLayer = 0; //determine the highest used layer
	BOOST_FOREACH(ptree::value_type &v, pt.get_child("layerTriplets"))
	{

		uint layer1 = v.second.get_child("layer1").get_value<uint>();
		uint layer2 = v.second.get_child("layer2").get_value<uint>();
		uint layer3 = v.second.get_child("layer3").get_value<uint>();

		float dThetaCut = v.second.get_child("dThetaCut").get_value<float>();
		float dThetaWindow = v.second.get_child("dThetaWindow").get_value<float>();
		float dPhiCut = v.second.get_child("dPhiCut").get_value<float>();
		float dPhiWindow = v.second.get_child("dPhiWindow").get_value<float>();
		uint pairSpreadZ = v.second.get_child("pairSpreadZ").get_value<uint>();
		uint pairSpreadPhi = v.second.get_child("pairSpreadPhi").get_value<uint>();
		float tipCut = v.second.get_child("tipCut").get_value<float>();

		if( (n == -1) || (added < n) ){
			//Layer1, Layer2, Layer3, dThetaCut, dThetaWindow, dPhiCut, dPhiWindow, tipCut, pairSpreadZ, pairSpreadPhi
			addWithValue(layer1, layer2, layer3, dThetaCut, dThetaWindow, dPhiCut, dPhiWindow, tipCut, pairSpreadZ, pairSpreadPhi);
			++added;

			if(layer3 > maxLayer)
				maxLayer = layer3;
		}
	}
	return maxLayer;

}



