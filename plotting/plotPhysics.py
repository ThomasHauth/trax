import csv
import numpy as np
import matplotlib.pyplot as plt

 # tt bar physics
ttbar = np.genfromtxt("physics.ttBar.reco.conf.csv", delimiter = " ", names=True)

# cmssw data
cmssw_org = np.genfromtxt("cmsswTiming.org.csv", delimiter = " ", names=True)
cmssw_org.sort(order='tracks')

cmssw_kd = np.genfromtxt("cmsswTiming.kd.csv", delimiter = " ", names=True)
cmssw_kd.sort(order='tracks')

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
#plot cmssw over tracks
plt.figure()
plt.plot(cmssw_org['tracks'], cmssw_org['eff'], 'go-', label='CMSSW efficiency')
plt.plot(cmssw_org['tracks'], cmssw_org['fr'], 'ro-', label='CMSSW fake rate')
plt.plot(cmssw_org['tracks'], cmssw_org['cr'], 'yo-', label='CMSSW clone rate')

#plt.plot(cmssw_kd['tracks'], cmssw_kd['eff'], 'go--', label=r'CMSSW $k$-d tree efficiency')
#plt.plot(cmssw_kd['tracks'], cmssw_kd['fr'], 'ro--', label=r'CMSSW $k$-d tree fake rate')
#plt.plot(cmssw_kd['tracks'], cmssw_kd['cr'], 'yo--', label=r'CMSSW $k$-d tree clone rate')

plt.title('Physics performance over number of tracks')
plt.xlabel('tracks')
plt.legend()

#******************************
# plot over eta

for data in etas:
    plt.figure()
    plt.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
    plt.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
    plt.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')
    
    plt.title(r'$t\bar{t}$ simulated event studies over $\eta$')
    plt.xlabel(r'$\eta$')
    plt.legend()

#******************************
# plot over eta

for data in pts:
    plt.figure()
    plt.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
    plt.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
    plt.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')
    
    plt.title(r'$t\bar{t}$ simulated event studies over $p_T$')
    plt.xlabel(r'$p_t$ in [$\frac{GeV}{c}$]')
    plt.xscale("log")
    plt.legend()

plt.show()
