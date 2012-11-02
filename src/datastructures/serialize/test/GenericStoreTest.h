#pragma once

#include "../Event.pb.h"
#include "../PEventStore.h"
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>

#include <boost/ptr_container/ptr_vector.hpp>

TEST( GenericStoreTest, read_write_mem )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	std::stringstream globalStream;

	typedef GenericInputStore< PEvent::PEvent, std::stringstream > PEventIStore;
	typedef GenericOutputStore< PEvent::PEvent, std::stringstream > PEventOStore;
	PEventOStore::WeakElementList elements;
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
	}
}
/*
TEST( GenericStoreTest, read_write_file )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	const std::string testFileName = "prototest.bin";
	const size_t hitCount = 20;
	//std::stringstream globalStream;

	std::ofstream outputFile(testFileName, std::ios::out | std::ios::binary | std::ios::trunc);

	typedef GenericInputStore< PEvent::PEvent, std::ifstream > PEventIStore;
	typedef GenericOutputStore< PEvent::PEvent, std::ofstream > PEventOStore;
	PEventOStore::WeakElementList elements;
	boost::ptr_vector< PEvent :: PEvent > strongElements;
	// going to 100000 fails
	const size_t iterations = 10000;
	const float fx = 300.2f;
	const float fy = -300.2f;
	const float fz = 0.2f;

	{
		/// VERY IMPORTANT: the PEventOStore must be destroyed ( and the containing proto buffers before trying to
		// read from the stream. Otherwise not all content will be written to the stream !
		// this is achieved via the nested C++ scope here

		PEventOStore eventOStore ( outputFile );

		for ( size_t i = 0; i < iterations; i ++)
		{
			PEvent :: PEvent * eventOne = new PEvent :: PEvent();

			eventOne->set_eventnumber( i + 1 );
			eventOne->set_lumisection(200);
			eventOne->set_runnumber(10000);

			// add some hits
			for (size_t j = 0; j < hitCount; j++)
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

			// free the mem regularly
			if ( ( i % 100 ) == 99 )
			{
				eventOStore.storeElements( elements );
				strongElements.clear();
				elements.clear();

			}
		}

		eventOStore.storeElements( elements );
		strongElements.clear();
		elements.clear();
	}

	outputFile.close();

	std::ifstream inputFile(testFileName, std::ios::in | std::ios::binary);

	//globalStream << "0000";
	//globalStream.flush();
	//globalStream.seekg( 0, std::ios::beg );

	//std::cout << "buffer size " << globalStream.str().size() << std::endl;

	PEventIStore eventIStore( inputFile );

	// read the first 10
	size_t readCount = eventIStore.readNextElements( 10, elements );
	ASSERT_EQ( size_t(10), readCount );
	ASSERT_EQ( size_t(10), elements.size() );

	ASSERT_EQ( uint64_t( 6), elements[5]->eventnumber() );
	ASSERT_EQ( uint64_t(10000), elements[5]->runnumber() );

	ASSERT_EQ( uint64_t(7), elements[6]->eventnumber() );
	ASSERT_EQ( fx, elements[6]->hits(hitCount-2).position().x() );
	ASSERT_EQ( fy, elements[6]->hits(hitCount-2).position().y() );
	ASSERT_EQ( fz, elements[6]->hits(hitCount-2).position().z() );

	// read the rest
	readCount = eventIStore.readNextElements( iterations - 10, elements );
	ASSERT_EQ( size_t( iterations - 10), readCount );

	ASSERT_EQ( size_t(iterations), elements.size() );

	for ( auto e : elements)
	{
		delete e;
	}
}

*/

