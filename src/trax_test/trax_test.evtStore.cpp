#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include <datastructures/serialize/Event.pb.h>
#include <datastructures/serialize/PEventStore.h>


using namespace std;
using namespace PB_Event;



int main(int argc, char **argv) {

	{
		EventStoreOutput out("test.dat");
		std::cout << "Opened file for writing with " << out.getEvents() << " events" << std::endl;

		for(uint i = 1; i < 10; ++i){
			PEvent event;
			event.set_eventnumber(i);
			event.set_lumisection(1);
			event.set_runnumber(1);

			for(uint j = 1; j < 100; ++j){
				PHit * pHit = event.add_hits();

				pHit->set_detectorid(23);
				pHit->set_layer(3);
				pHit->set_hitid( j );
				pHit->set_detectortype(DetectorType::BARREL);

				pHit->mutable_position()->set_x(j);
				pHit->mutable_position()->set_y(j);
				pHit->mutable_position()->set_z(j);
			}

			out.storeElement(event);

			std::cout << "Written " << i << std::endl;
		}

		std::cout << "Written events: " << out.getEvents();

		out.Close();

		std::cout<< " after close: " << out.getEvents() << std::endl;

		EventStoreInput in("test.dat");

		std::cout << "Opened file with " << in.getEvents() << " events" << std::endl;

		PEvent * event = in.readNextElement();
		while(event != NULL){
			std::cout << "read event " << event->eventnumber() << std::endl;
			event = in.readNextElement();
		}

		in.Close();
	}

	for(uint j = 1; j < 100; ++j)
	{
		EventStoreOutput out("test.dat", true);
		std::cout << "Opened file for writing with " << out.getEvents() << " events" << std::endl;

		uint n = out.getEvents();
		for(uint i = n+1; i < n + 10; ++i){
			PEvent event;
			event.set_eventnumber(i);
			event.set_lumisection(1);
			event.set_runnumber(1);

			for(uint j = 1; j < 100; ++j){
				PHit * pHit = event.add_hits();

				pHit->set_detectorid(23);
				pHit->set_layer(3);
				pHit->set_hitid( j );
				pHit->set_detectortype(DetectorType::BARREL);

				pHit->mutable_position()->set_x(j);
				pHit->mutable_position()->set_y(j);
				pHit->mutable_position()->set_z(j);
			}

			out.storeElement(event);

			std::cout << "Written " << i << std::endl;
		}

		std::cout << "Written events: " << out.getEvents();

		out.Close();

		std::cout<< " written: " << out.getEvents() << std::endl;

		EventStoreInput in("test.dat");

		std::cout << "Opened file with " << in.getEvents() << " events" << std::endl;

		PEvent * event = in.readNextElement();
		while(event != NULL){
			std::cout << "read event " << event->eventnumber() << std::endl;
			event = in.readNextElement();
		}

		in.Close();
	}
}



/*PEventOStore::WeakElementList elements;
	boost::ptr_vector< PEvent :: PEvent > strongElements;
	const size_t iterations = 1000;
	const float fx = 300.2f;
	const float fy = -300.2f;
	const float fz = 0.2f;

	{
		/// VERY IMPORTANT: the PEventOStore must be destroyed ( and the containing proto buffers before trying to
		// read from the stream. Otherwise not all content will be written to the stream !
		// this is achieved via the nested C++ scope here
		PEventOStore eventOStore ( globalStream );

		for ( size_t i = 0; i < iterations; i ++)
		{
			PEvent :: PEvent * eventOne = new PEvent :: PEvent();

			eventOne->set_eventnumber( i + 1 );
			eventOne->set_lumisection(200);
			eventOne->set_runnumber(10000);

			// add some hits
			for (size_t j = 0; j < 1000; j++)
			{

				PEvent ::PHit * pHit = eventOne->add_hits();

				pHit->set_detectorid(23);
				pHit->set_layer(3);
				pHit->set_hitid( j );

				pHit->mutable_position()->set_x(fx);
				pHit->mutable_position()->set_y(fy);
				pHit->mutable_position()->set_z(fz);
			}
			strongElements.push_back ( eventOne );
			elements.push_back ( eventOne );
		}

		eventOStore.storeElements( elements );
	}

	strongElements.clear();
	elements.clear();

	globalStream.seekg( 0, std::ios::beg );

	PEventIStore eventIStore( globalStream );

	// read the first 10
	size_t readCount = eventIStore.readNextElements( 10, elements );
	ASSERT_EQ( size_t(10), readCount );
	ASSERT_EQ( size_t(10), elements.size() );

	ASSERT_EQ( uint64_t( 6), elements[5]->eventnumber() );
	ASSERT_EQ( uint64_t(10000), elements[5]->runnumber() );

	ASSERT_EQ( uint64_t(7), elements[6]->eventnumber() );
	ASSERT_EQ( fx, elements[6]->hits(233).position().x() );
	ASSERT_EQ( fy, elements[6]->hits(233).position().y() );
	ASSERT_EQ( fz, elements[6]->hits(233).position().z() );

	// read the rest
	readCount = eventIStore.readNextElements( iterations - 10, elements );
	ASSERT_EQ( size_t( iterations - 10), readCount );

	ASSERT_EQ( size_t(iterations), elements.size() );

	for ( auto e : elements)
	{
		delete e;
	}*/
