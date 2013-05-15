#include "GridBuilder.h"

#include <iostream>
#include <iomanip>

cl_ulong GridBuilder::run(HitCollection & hits, uint nThreads, const EventSupplement & eventSupplement, const LayerSupplement & layerSupplement, Grid & grid)
{

	//store buffers to input data
	cl_mem x = hits.transfer.buffer(GlobalX()); cl_mem y =  hits.transfer.buffer(GlobalY()); cl_mem z =  hits.transfer.buffer(GlobalZ());
	cl_mem layer= hits.transfer.buffer(DetectorLayer()); cl_mem detId = hits.transfer.buffer(DetectorId()); cl_mem hitId = hits.transfer.buffer(HitId()); cl_mem evtId = hits.transfer.buffer(EventNumber());

	std::vector<cl_event> events;

	LOG << "Grid count kernel" << std::endl;
	cl_event evt = gridBuilding_kernel.run(
			//configuration
			eventSupplement.transfer.buffer(Offset()), layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()), layerSupplement.nLayers,
			//grid data
			grid.transfer.buffer(Boundary()),
			//grid configuration
			grid.config.MIN_Z, grid.config.sectorSizeZ(), grid.config.nSectorsZ,
			grid.config.MIN_PHI, grid.config.sectorSizePhi(),grid.config.nSectorsPhi,
			// hit input
			x, y, z,
			//local memory
			local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
			//work item config
			range(nThreads,layerSupplement.nLayers, eventSupplement.nEvents),
			range(nThreads,1, 1));
	events.push_back(evt);

	//run prefix sum on grid boundaries
	LOG << "Grid prefix kernel" << std::endl;
	PrefixSum prefixSum(ctx);

	evt = prefixSum.run(grid.transfer.buffer(Boundary()), grid.size(),  nThreads);
	events.push_back(evt);

	//download updated grid
	grid.transfer.fromDevice(ctx,grid, &events);
	if(PROLIX){
		printGrid(grid);
	}

	//allocate new buffers for ouput
	hits.transfer.initBuffers(ctx, hits);

	LOG << "Grid store kernel" << std::endl;
	evt = gridStore_kernel.run(
			//configuration
			eventSupplement.transfer.buffer(Offset()), layerSupplement.transfer.buffer(NHits()), layerSupplement.transfer.buffer(Offset()), grid.config.nLayers,
			//grid data
			grid.transfer.buffer(Boundary()),
			//grid configuration
			grid.config.MIN_Z, grid.config.sectorSizeZ(), grid.config.nSectorsZ,
			grid.config.MIN_PHI, grid.config.sectorSizePhi(),grid.config.nSectorsPhi,
			// hit input
			x,y,z,
			layer,detId, hitId, evtId,
			// hit output
			hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
			hits.transfer.buffer(DetectorLayer()), hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()), hits.transfer.buffer(EventNumber()),
			//local params
			local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)), local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
			//work item config
			range(nThreads,layerSupplement.nLayers, eventSupplement.nEvents),
			range(nThreads,1, 1));

	events.push_back(evt);

	//download updated hit data and grid
	//grid.transfer.fromDevice(ctx,grid, &events);
	//printGrid(grid);
	hits.transfer.fromDevice(ctx,hits, &events);
	if(PROLIX)
		verifyGrid(hits, grid);

	//delete old buffers
	ctx.release_buffer(x); ctx.release_buffer(y); ctx.release_buffer(z);
	ctx.release_buffer(layer); ctx.release_buffer(detId); ctx.release_buffer(hitId); ctx.release_buffer(evtId);

	return 0;
}

void GridBuilder::printGrid(const Grid & grid){
	for(uint e = 0; e < grid.config.nEvents; ++e){
		PLOG << "Event: " << e << std::endl;
		for(uint l = 1; l <= grid.config.nLayers; ++l){
			PLOG << "Layer: " << l << std::endl;

			//output z boundaries
			PLOG << "z/phi\t\t";
			for(uint i = 0; i <= grid.config.nSectorsZ; ++i){
				PLOG << grid.config.boundaryValueZ(i) << "\t";
			}
			PLOG << std::endl;

			LayerGrid layerGrid(grid, l,e);
			for(uint p = 0; p <= grid.config.nSectorsPhi; ++p){
				PLOG << std::setprecision(3) << grid.config.boundaryValuePhi(p) << "\t\t";
				for(uint z = 0; z <= grid.config.nSectorsZ; ++z){
					PLOG << layerGrid(z,p) << "\t";
				}
				PLOG << std::endl;
			}
		}
	}
}

void GridBuilder::verifyGrid(HitCollection & hits, const Grid & grid){

	bool valid = true;
	for(uint e = 0; e < grid.config.nEvents; ++e){
		for(uint l = 1; l <= grid.config.nLayers; ++l){

			LayerGrid layerGrid(grid, l,e);
			for(uint z = 0; z < grid.config.nSectorsZ; ++z){
				for(uint p = 0; p < grid.config.nSectorsPhi; ++p){
					for(uint h = layerGrid(z, p); h < layerGrid(z,p+1); ++h){
						Hit hit(hits, h);
						if(hit.getValue<EventNumber>() % grid.config.nEvents != e){
							/* ist not necessarly wrong -> skip events
							valid = false;
							PLOG << "Invalid event number" << std::endl;*/
						}
						if(hit.getValue<DetectorLayer>() != l){
							valid = false;
							PLOG << "Invalid layer " << std::endl;
						}
						if(hit.globalZ() < grid.config.boundaryValueZ(z) || hit.globalZ() > grid.config.boundaryValueZ(z+1)){
							valid = false;
							PLOG << "Invalid z cell" << std::endl;
						}
						if(hit.phi() < grid.config.boundaryValuePhi(p) || hit.phi() > grid.config.boundaryValuePhi(p+1)){
							valid = false;
							PLOG << "Invalid phi cell" << std::endl;
						}
					}

				}
			}
		}
	}

	if(valid)
		LOG << "Grid built correctly" << std::endl;
}
