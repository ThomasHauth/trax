import csv
import numpy as np
import matplotlib.pyplot as plt

 # tt bar physics
ttbar = np.genfromtxt("physics.ttBar.reco.conf.csv", delimiter = " ", names=True)

# load eta and pt histograms
etas = []
pts = []
for lt in ttbar['layerTriplet']:
    etas.append(np.genfromtxt("eta_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    pts.append(np.genfromtxt("pt_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    
#******************************
# overview plot

plt.figure()
plt.errorbar(ttbar['layerTriplet']+1, ttbar['eff'], yerr=ttbar['effVar'], fmt='go-', label='efficiency')
plt.errorbar(ttbar['layerTriplet']+1, ttbar['fr'], yerr=ttbar['frVar'], fmt='ro-', label='fake rate')
plt.errorbar(ttbar['layerTriplet']+1, ttbar['cr'], yerr=ttbar['crVar'], fmt='yo-', label='clone rate')

plt.title(r'$t\bar{t}$ simulated event studies')
plt.xlabel("layer triplet")
plt.legend()

#******************************
# plot over eta

data = etas[0]

plt.figure()
plt.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
plt.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
plt.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')

plt.title(r'$t\bar{t}$ simulated event studies over $\eta$')
plt.xlabel(r'$\eta$')
plt.legend()

#******************************
# plot over eta

data = pts[0]

plt.figure()
plt.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
plt.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
plt.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')

plt.title(r'$t\bar{t}$ simulated event studies over $p_T$')
plt.xlabel(r'$p_t$ in [$\frac{GeV}{c}$]')
plt.xscale("log")
plt.legend()

plt.show()
