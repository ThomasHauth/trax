#include "HitCollection.h"

#include "serialize/Event.pb.h"

HitCollection::HitCollection(std::vector<PEvent::PEvent *> events)
{
	for ( auto e : events )
	{
		for ( auto hit: e->hits() )
		{
			addWithValue( hit.position().x(),  hit.position().y(),  hit.position().z(),
							hit.detectorid(), hit.hitid(),
							e->eventnumber());
		}
	}

}
