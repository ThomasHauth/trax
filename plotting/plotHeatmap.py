import csv
import numpy as np
import matplotlib.pyplot as plt
import Event_pb2
import math

eventContainer = Event_pb2.PEventContainer()

# Read the existing address book.
f = open('../data/hitsPXB_TIB_TOB.pb', "rb")
eventContainer.ParseFromString(f.read())
f.close()

x = []
y = []

print len(eventContainer.events)

for event in eventContainer.events:
    
    print len(event.hits)
    for hit in event.hits:
        phi = math.atan2(hit.position.x, hit.position.y)
        y.append(phi)
        x.append(hit.position.z)

#print x, y
gridsize=30

# if 'bins=None', then color of each hexagon corresponds directly to its count
# 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# the result is a pure 2D histogram 

plt.figure()
plt.hexbin(x, y, C=None, gridsize=gridsize, bins=None)
plt.axis([-100, 100, -3.14, 3.14])

cb = plt.colorbar()
cb.set_label('entries')
plt.show()   