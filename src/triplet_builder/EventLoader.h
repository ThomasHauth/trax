#pragma once

#include <iostream>
#include <fcntl.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <datastructures/serialize/Event.pb.h>
#include <datastructures/Logger.h>

#include "Parameters.h"

class EventLoader{

public:
	EventLoader(EventDataLoadingParameters config ){
		params = config;

		int fd = open(config.eventDataFile.c_str(), O_RDONLY);
		google::protobuf::io::FileInputStream fStream(fd);
		google::protobuf::io::CodedInputStream cStream(&fStream);

		cStream.SetTotalBytesLimit(536870912, -1);

		if(!pContainer.ParseFromCodedStream(&cStream)){
			std::cerr << "Could not read protocol buffer" << std::endl;
			return;
		} else {
			LOG << "Opened file " << config.eventDataFile << " with " << pContainer.events_size() << " events" << std::endl;
		}
		cStream.~CodedInputStream();
		fStream.Close();
		fStream.~FileInputStream();
		close(fd);

	}

	virtual ~EventLoader(){}

	virtual int nEvents() const {
		return pContainer.events_size();
	}

	virtual const PB_Event::PEvent & getEvent(uint e) const {
		return pContainer.events(e);
	}


protected:
	EventLoader() { }

	EventDataLoadingParameters params;

private:
	PB_Event::PEventContainer pContainer;
};

class RepeatedEventLoader : public EventLoader{

public:
	RepeatedEventLoader(EventDataLoadingParameters config ){
		params = config;

		int fd = open(config.eventDataFile.c_str(), O_RDONLY);
		google::protobuf::io::FileInputStream fStream(fd);
		google::protobuf::io::CodedInputStream cStream(&fStream);

		cStream.SetTotalBytesLimit(536870912, -1);

		PB_Event::PEventContainer pContainer;
		if(!pContainer.ParseFromCodedStream(&cStream)){
			std::cerr << "Could not read protocol buffer" << std::endl;
			return;
		} else {
			LOG << "Opened file " << config.eventDataFile << " with " << pContainer.events_size() << " events" << std::endl;
		}
		cStream.~CodedInputStream();
		fStream.Close();
		fStream.~FileInputStream();
		close(fd);

		event = pContainer.events(0);

	}

	virtual ~RepeatedEventLoader(){ }

	virtual int nEvents() const {
		return params.maxEvents;
	}

	virtual const PB_Event::PEvent & getEvent(uint e) const {
		return event;
	}


private:
	PB_Event::PEvent event;
};
