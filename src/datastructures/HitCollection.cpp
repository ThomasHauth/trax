#include "HitCollection.h"

#include "serialize/Event.pb.h"

HitCollection::HitCollection(const std::vector<PB_Event::PEvent *>& events)
{
	for ( auto e : events )
	{
		addEvent(*e);
	}

}

HitCollection::HitCollection(const PB_Event::PEvent & event)
{
	addEvent(event);
}
