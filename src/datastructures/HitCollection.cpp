#include "HitCollection.h"

#include "serialize/Event.pb.h"
#include "LayerSupplement.h"

HitCollection::HitCollection(const std::vector<PB_Event::PEvent *>& events)
{
	for ( auto e : events )
	{
		//addEvent(*e, DetectorGeometry(), LayerSupplement(0 , 0));
	}

}

HitCollection::HitCollection(const PB_Event::PEvent & event)
{
	//addEvent(event, DetectorGeometry(), LayerSupplement(0, 0));
}

HitCollection::tTrackList HitCollection::addEvent(const PB_Event::PEvent& event, const DetectorGeometry & geom, EventSupplement & eventSupplement, uint evtInGroup, LayerSupplement & layerSupplement , float minPt,
		int numTracks, bool onlyTracks, uint maxLayer) {

	//associate hits to tracks
	tTrackList allTracks;
	for (auto& hit : event.hits()) {

		if(onlyTracks && hit.simtrackid() == 0)
			continue;

		if (hit.simtrackpt() < minPt)
			continue;

		allTracks[hit.simtrackid()].push_back(hit);
	}

	//sort hits of tracks by layer
	for(auto & track : allTracks)
		std::sort(track.second.begin(), track.second.end(),
				[] (const PB_Event::PHit & a, const PB_Event::PHit & b)
				{
					return a.layer() < b.layer();
				});

	tTrackList layers; // hits bucket sorted into layers
	tTrackList findableTracks; //findable tracks = collection of tracks that count towards efficiency
	int foundTracks = 0; //found tracks initialized to 0; if no limit on track, i.e. numTracks=-1, found track will never be equal to numTracks
	for(tTrackList::const_iterator itTrack = allTracks.begin(); itTrack != allTracks.end() && foundTracks != numTracks; ++itTrack){
		//skip unassociated hits
		if(onlyTracks && itTrack->first == 0)
			continue;

		//check for missing hits
		bool missingHits = false;
		uint covLayers = 0;
		for(auto hit : itTrack->second){
			if(hit.layer() == (covLayers + 1)){
				++covLayers;
			}
		}
		if(covLayers < maxLayer)
			missingHits = true;

		//skip tracks with to low pt
		if(itTrack->second[0].simtrackpt() < minPt)
			continue;

		//add all hits to track
		for(auto & hit : itTrack->second){
			if(!missingHits)
				findableTracks[itTrack->first].push_back(hit);
			else if(!onlyTracks)
				layers[hit.layer()].push_back(hit);
		}

		if(!missingHits)
			++foundTracks;
	}

	//bucket sort findable trakc hits into layers
	for(tTrackList::const_iterator itTrack = findableTracks.begin(); itTrack != findableTracks.end(); ++itTrack){

		for (auto& hit : itTrack->second) {
			if(hit.layer() <= maxLayer){
				layers[hit.layer()].push_back(hit);
			}
		}
	}

	//fill event and layer supplement
	uint offset = 0;
	for(uint i = 1; i <= maxLayer; ++i){
		layerSupplement[evtInGroup * maxLayer + i-1].setNHits(layers[i].size());
		layerSupplement[evtInGroup * maxLayer + i-1].setOffset(offset);
		offset += layers[i].size();
	}
	//offset contains total number of hits for this event
	eventSupplement[evtInGroup].setNHits(offset);
	//use offset for evt offset computation
	offset = evtInGroup > 0 ? eventSupplement[evtInGroup-1].getOffset() + eventSupplement[evtInGroup-1].getNHits() : 0;
	eventSupplement[evtInGroup].setOffset(offset);


	//add hits in order of layers to HitCollection
	for(uint i = 1; i <= maxLayer; ++i){


		PLOG << "Layer " << i << std::endl;


		for(auto & hit : layers[i]){

			uint detId = hit.detectorid();
		    int index = geom.resolveDetId(detId);
		    if(index != -1)
		    	detId = index;
		    else
		    	PLOG << "Invalid DetectorId used: << " << detId << std::endl;

			int id = addWithValue(hit.position().x(), hit.position().y(),
						 	 	   hit.position().z(), hit.layer(), detId,
						 	 	   hit.simtrackid(), event.eventnumber());


			PLOG << "\t [" << id << "] Track: " << hit.simtrackid();
			PLOG << "\t\t [" << hit.position().x()<< ", " <<  hit.position().y()<< ", " <<  hit.position().z() << "]";
			PLOG << std::endl;

		}
	}

	return findableTracks;
}

HitCollection PHitWrapper::pHitCollection;
