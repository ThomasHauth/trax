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

void HitCollection::addEvent(const PB_Event::PEvent& event, float minPt, int numTracks){

	tTrackList tracks;
	for ( auto & hit: event.hits() )
	{

		if(hit.simtrackpt() < minPt)
			continue;

		auto track = tracks.find(hit.simtrackid());

		if(track != tracks.end())
			track->second.push_back(hit);
		else {
			tTrackHits trackHits;
			trackHits.push_back(hit);
			tracks.insert(tTrackListEntry(hit.simtrackid(), trackHits));
		}
	}

	int skip = numTracks != -1 ? tracks.size() / numTracks : 1;

		for(uint i = 1; i < tracks.size(); ++i){
		if((i % skip) == 0){
			for(auto & hit : tracks[i]){
				addWithValue( hit.position().x(),  hit.position().y(),  hit.position().z(),
						hit.layer(), hit.detectorid(), hit.hitid(),
						event.eventnumber());
				std::cout << hit.position().x()<< " | " <<  hit.position().y()<< " | " <<  hit.position().z()<< " | " <<
										hit.layer()<< " | " << hit.detectorid()<< " | " << hit.hitid()<< " | " <<
										event.eventnumber() << " | " << hit.simtrackid() << " | " << hit.simtrackpt() << std::endl;
			}
		}
	}

}
