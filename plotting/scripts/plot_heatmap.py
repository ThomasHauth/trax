import csv
import numpy as np

import latexHeader

import matplotlib.pyplot as plt
import matplotlib.colors as co
import Event_pb2
import math

eventContainer = Event_pb2.PEventContainer()

# Read the existing address book.
f = open('../data/physics/hits.ttBar.pb', "rb")
eventContainer.ParseFromString(f.read())
f.close()

phiPXB = []
zPXB = []

phiTIB = []
zTIB = []

#print len(eventContainer.events)

for event in eventContainer.events:
    
    #print len(event.hits)
    for hit in event.hits:
        phi = math.atan2(hit.position.x, hit.position.y)
        
        if hit.layer <= 3:
            phiPXB.append(phi)
            zPXB.append(hit.position.z)
            
        if hit.layer > 3 & hit.layer < 8:
            phiTIB.append(phi)
            zTIB.append(hit.position.z)


######################################################
### PXB

plt.figure()
plt.hexbin(zPXB, phiPXB, C=None, gridsize=(int(6.28/0.1), int(50/1)), bins=None)

plt.title('Hit Distribution in the PXB')

plt.ylim(-3.14, 3.14)
plt.xlim(-25, 25)

plt.xlabel('$z$')
plt.ylabel('$\phi$')

cb = plt.colorbar()
cb.set_label('hits/bin')

plt.gcf().text(0.12, 0.905, r'10 $t\bar{t}$ events', va='bottom', ha='left')

plt.savefig("../output/physics/heatmap_PXB.pdf")

######################################################
### TIB

plt.figure()
plt.hexbin(zTIB, phiTIB, C=None, gridsize=(int(6.28/0.1), int(200/5)), bins=None)
plt.ylim(-3.14, 3.14)
plt.xlim(-100, 100)

plt.title('Hit Distribution in the TIB')

plt.xlabel('$z$')
plt.ylabel('$\phi$')

cb = plt.colorbar()
cb.set_label('hits/bin')

plt.gcf().text(0.12, 0.905, r'10 $t\bar{t}$ events', va='bottom', ha='left')

plt.savefig("../output/physics/heatmap_TIB.pdf")  