/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <iomanip>
#include <set>
#include <fcntl.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <datastructures/serialize/Event.pb.h>
#include <datastructures/serialize/PEventStore.h>

#include "EventLoader.h"


int main(int argc, char *argv[]) {

	namespace po = boost::program_options;

	std::string inputFile;
	std::string outputFile;
	uint skipEvents;
	int maxEvents;
	uint multiplicity;

	po::options_description cCommandLine("Command Line Options");
	cCommandLine.add_options()
		("inputFile", po::value<std::string>(&inputFile), "event store file file")
		("outputFile", po::value<std::string>(&outputFile), "outputFile file")
		("skipEvents", po::value<uint>(&skipEvents)->default_value(0), "skip events, default 0")
		("maxEvents", po::value<int>(&maxEvents)->default_value(-1), "max events, default -1 [all]")
		("multiplicity", po::value<uint>(&multiplicity)->default_value(1), "multiply events, default 1 [no]")
		("help", "produce help message")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,cCommandLine), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << cCommandLine << std::endl;
		return 1;
	}

	EventStoreInput esIn(inputFile);

	//apply skip events
	for(uint i = 0; i < skipEvents; ++i)
		esIn.readNextElement();

	uint lastEvent = 0;
	if(maxEvents == -1)
		lastEvent = esIn.getEvents();
	else
		lastEvent = std::min((uint) maxEvents, esIn.getEvents());

	PB_Event::PEventContainer container;

	for(uint i = skipEvents; i < lastEvent; ++i){

		PB_Event::PEvent * evt =  esIn.readNextElement();

		for(uint j = 0; j < multiplicity; ++j){
			PB_Event::PEvent * newEvt = container.add_events();
			newEvt->CopyFrom(*evt);
		}
	}

	if(!vm.count("outputFile"))
		outputFile = "container.pb";

	std::fstream out(outputFile, std::ios::out | std::ios::binary | std::ios::trunc);
	container.SerializeToOstream( &out );
	out.close();

}
