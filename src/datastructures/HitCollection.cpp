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

HitCollection::tTrackList HitCollection::addEvent(const PB_Event::PEvent& event, const DetectorGeometry & geom, LayerSupplement & layerSupplement , float minPt,
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

	tTrackList filteredTracks;
	int foundTracks = 0; //found tracks initialized to 0; if no limit on track, i.e. numTracks=-1, found track will never be equal to numTracks
	for(tTrackList::const_iterator itTrack = allTracks.begin(); itTrack != allTracks.end() && foundTracks != numTracks; ++itTrack){
		//skip unassociated hits
		if(onlyTracks && itTrack->first == 0)
			continue;

		//TODO[gpu] REMOVE again - intermediate fix for singleMu
		if(onlyTracks && itTrack->first == 1)
					continue;

		//skip tracks with missing hits
		if(onlyTracks){
			uint covLayers = 0;
			for(auto hit : itTrack->second){
				if(hit.layer() == (covLayers + 1)){
					++covLayers;
				}
			}
			if(covLayers < maxLayer)
				continue;
		}
		//skip tracks with to low pt
		if(itTrack->second[0].simtrackpt() < minPt)
			continue;

		//add all hits to track
		for(auto & hit : itTrack->second)
			filteredTracks[itTrack->first].push_back(hit);

		++foundTracks;
	}

	//bucket sort hits into layers
	tTrackList layers;
	for(tTrackList::const_iterator itTrack = filteredTracks.begin(); itTrack != filteredTracks.end(); ++itTrack){

		for (auto& hit : itTrack->second) {
			if(hit.layer() <= maxLayer){
				layers[hit.layer()].push_back(hit);
			}
		}
	}

	//fill layer sizes
	if(layerSupplement.size() > 0){
		uint offset = 0;
		for(uint i = 1; i <= maxLayer; ++i){
			layerSupplement[i-1].setNHits(layers[i].size());
			layerSupplement[i-1].setOffset(offset);
			offset += layers[i].size();
		}
	}

#define OUT_BY_LAYER
	//add hits in order of layers to HitCollection
	for(uint i = 1; i <= maxLayer; ++i){

#ifdef OUT_BY_LAYER
		std::cout << "Layer " << i << std::endl;
#endif

		for(auto & hit : layers[i]){

			uint detId = hit.detectorid();
		    int index = geom.resolveDetId(detId);
		    if(index != -1)
		    	detId = index;
		    else
		    	std::cerr << "Invalid DetectorId used: << " << detId << std::endl;

			int id = addWithValue(hit.position().x(), hit.position().y(),
						 	 	   hit.position().z(), hit.layer(), detId,
						 	 	   hit.simtrackid(), event.eventnumber());

#ifdef OUT_BY_LAYER
			std::cout << "\t [" << id << "] Track: " << hit.simtrackid();
			std::cout << "\t\t [" << hit.position().x()<< ", " <<  hit.position().y()<< ", " <<  hit.position().z() << "]";
			std::cout << std::endl;
#endif
		}
	}

	return filteredTracks;
}
