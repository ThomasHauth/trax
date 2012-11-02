#include "EventProcessor.h"

#include <datastructures/HitCollection.h>
#include <algorithms/DummyTracklets.h>

EventProcessor::EventProcessor(std::string const& inputFilename,
		clever::context * openClContext, int maxEvents, size_t concurrentEvents) :
		m_inputFileName(inputFilename), m_openCLContext(openClContext), m_maxEvents(
				maxEvents), m_concurrentEvents(concurrentEvents)
{
	assert( m_concurrentEvents > 0);
	m_algoDummyTracklets.reset(new DummyTracklets(*m_openCLContext));
}

EventProcessor::~EventProcessor()
{
}

void EventProcessor::run()
{
	std::ifstream inputFile(m_inputFileName, std::ios::in | std::ios::binary);
	EventStoreInput evInput(inputFile);

	// could be made multithreaded: one thread loading events, the other one processing

	// always buffer at least 10 events
	size_t loadEvents = std::max(size_t(10), m_concurrentEvents);
	// also don't load more then the max events into the buffer
	if (m_maxEvents > -1)
		loadEvents = std::min(loadEvents, size_t(m_maxEvents));

	EventStoreInput::WeakElementList bufferedEvents;

	size_t numberRead = loadEvents;
	int eventsProcessed = 0;

	// this loop will end once less events then requested were loaded -> end of file
	while (numberRead == loadEvents)
	{
		numberRead = evInput.readNextElements(loadEvents, bufferedEvents);

		std::cout << "Loaded " << numberRead << " into buffer" << std::endl;
		if (numberRead == 0)
			break;
		//process this

		for (size_t i = 0; i < m_concurrentEvents; i = +m_concurrentEvents)
		{
			EventStoreInput::WeakElementList eventSet;
			auto it_local_begin = bufferedEvents.begin();

			for (auto it_local_current = it_local_begin;
					it_local_current != (it_local_begin + m_concurrentEvents);
					it_local_current++)
			{
				eventSet.push_back(*it_local_current);
				eventsProcessed++;
				if (eventsProcessed == m_maxEvents)
					break;
			}
			processSubset(eventSet);

			if (eventsProcessed == m_maxEvents)
				break;
		}

		// delete events in bufferedEvents
		// could also use ptr_vector here
		for (auto e : bufferedEvents)
		{
			delete e;
		}
		bufferedEvents.clear();

		if (eventsProcessed == m_maxEvents)
			break;
	}
	//for ( )

	std::cout << "All " << eventsProcessed << " event(s) processed"
			<< std::endl;
}

void EventProcessor::processSubset(
		EventStoreInput::WeakElementList & eventSubset)
{
	std::cout << "Starting process event subset of " << eventSubset.size()
			<< std::endl;

	// this can handle at max 5 par-events with 20 hits each ...
	TrackletCollection tracklets(10000000);
	// get all the input data ready
	HitCollection hits(eventSubset);

	//get transfer ready
	// todo: dont create the bufferst each time, keep them
	HitCollectionTransfer clTrans;
	TrackletCollectionTransfer clTrans_tracklet;

	clTrans.initBuffers(*m_openCLContext, hits);
	clTrans.toDevice(*m_openCLContext, hits);

	clTrans_tracklet.initBuffers(*m_openCLContext, tracklets);

	// run kernel
	unsigned int found = m_algoDummyTracklets->run(clTrans, clTrans_tracklet);

	clTrans_tracklet.fromDevice(*m_openCLContext, tracklets);

	std::cout << found << " tracklets found" << std::endl;

}
