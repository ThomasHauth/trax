#pragma once

#include "../Event.pb.h"
#include "../PEventStore.h"
#include <sstream>
#include <gtest/gtest.h>

TEST( EventSerializationTest, read_write )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	PB_Event :: PEvent eventOne;

	const float fx = 300.2f;
	const float fy = -300.2f;
	const float fz = 0.2f;

	eventOne.set_eventnumber(23);
	eventOne.set_lumisection(200);
	eventOne.set_runnumber(10000);

	for (size_t j = 0; j < 10; j++)
	{

		PB_Event ::PHit * pHit = eventOne.add_hits();

		pHit->set_detectorid(23);
		pHit->set_layer(3);
		pHit->set_hitid( j );

		pHit->mutable_position()->set_x(fx);
		pHit->mutable_position()->set_y(fy);
		pHit->mutable_position()->set_z(fz);
	}

	const std::string serString(eventOne.SerializeAsString());

	PB_Event::PEvent eventNew;

	eventNew.ParseFromString( serString );

	ASSERT_EQ( 10,eventNew.hits_size() );
	ASSERT_EQ( uint64_t( 23 ),eventNew.eventnumber() );

	ASSERT_EQ( fx,eventNew.hits(0).position().x() );
	ASSERT_EQ( fy,eventNew.hits(0).position().y() );
	ASSERT_EQ( fz,eventNew.hits(0).position().z() );

}

TEST( EventSerializationTest, speed_test )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	std::stringstream tempString;

	// works, but this has to be implemented
	// to make in seriously work
	// https://groups.google.com/forum/?fromgroups=#!topic/protobuf/A4zErQALQmU

	const size_t iterations = 1000;
	const float fx = 300.2f;
	const float fy = -300.2f;
	const float fz = 0.2f;

	PB_Event::PEventContainer cont;

	for ( size_t i = 0; i < iterations; i ++)
	{

		PB_Event :: PEvent * eventOne = cont.add_events();

		eventOne->set_eventnumber( i + 2);
		eventOne->set_lumisection(200);
		eventOne->set_runnumber(10000);

		// add some hits
		for (size_t j = 0; j < 1000; j++)
		{

			PB_Event ::PHit * pHit = eventOne->add_hits();

			pHit->set_detectorid(23);
			pHit->set_layer(3);
			pHit->set_hitid( j );

			pHit->mutable_position()->set_x(fx);
			pHit->mutable_position()->set_y(fy);
			pHit->mutable_position()->set_z(fz);
		}
	}

	cont.SerializeToOstream( &tempString );

	PB_Event::PEventContainer contNew;
	contNew.ParseFromIstream( &tempString );
}
