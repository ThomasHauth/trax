#pragma once

#include <memory>
#include <algorithm>
#include <fstream>
#include <string>

#include <clever/clever.hpp>

#include <boost/noncopyable.hpp>


#include <datastructures/serialize/PEventStore.h>


class DummyTracklets;

class EventProcessor: private boost::noncopyable
{
public:
	EventProcessor(std::string const& inputFilename, int maxEvents = -1,
			size_t concurrentEvents = 1);

	~EventProcessor();

	void run() ;

	void processSubset(EventStoreInput::WeakElementList & eventSubset) ;

private:
	const std::string m_inputFileName;
	const int m_maxEvents;
	const size_t m_concurrentEvents;
	clever::context m_openCLContext;
	std::unique_ptr < DummyTracklets > m_algoDummyTracklets;

};
