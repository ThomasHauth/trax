import csv
import numpy as np

import matplotlib as mpl

mpl.use('pgf')
latexConf = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{siunitx}",         # load additional packages
    ]}
mpl.rcParams.update(latexConf)

import matplotlib.pyplot as plt

 # tt bar physics
ttbar = np.genfromtxt("physics.ttBar.reco.conf.csv", delimiter = " ", names=True)

# cmssw data
cmssw_org = np.genfromtxt("cmsswTiming.org.csv", delimiter = " ", names=True)
cmssw_org.sort(order='tracks')

cmssw_kd = np.genfromtxt("cmsswTiming.kd.csv", delimiter = " ", names=True)
cmssw_kd.sort(order='tracks')

#cmssw physics comparison
physicsComparison = np.genfromtxt("physicsCMSSW.csv", delimiter = " ", names=True)

#define layer triplets
layerTriplets = ["", "1-2-3", "2-3-4", "3-4-5", "4-5-8", "", "combined"]
overview = 5

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
plt.xlim(0, overview+2)
plt.gca().set_xticklabels(layerTriplets)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)

plt.savefig('physics_overview.pdf')

#******************************
#plot cmssw over tracks
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))

ax.plot(physicsComparison['tracks'], physicsComparison['eff'], 'go-', label=r'efficiency')
ax.plot(physicsComparison['tracks'], physicsComparison['fr'], 'ro-', label=r'fake rate')
ax.plot(physicsComparison['tracks'], physicsComparison['cr'], 'yo-', label=r'clone rate')

ax.plot(cmssw_org['tracks'], cmssw_org['eff'], 'gx--', label='CMSSW efficiency')
ax.plot(cmssw_org['tracks'], cmssw_org['fr'], 'rx--', label='CMSSW fake rate')
ax.plot(cmssw_org['tracks'], cmssw_org['cr'], 'yx--', label='CMSSW clone rate')

ax.set_title('Physics performance over number of tracks')
ax.set_xlabel('tracks')
ax.set_xscale('log')
ax.set_ylim(0, 1.1)
ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure, loc=5)

plt.savefig('physics_cmssw.pdf')

#******************************
# plot over eta

for data, lt in zip(etas, ttbar['layerTriplet']):
    fig, ax = plt.subplots()
    ax.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
    ax.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
    ax.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')
    
    ax.set_title(r'$t\bar{t}$ simulated event studies over $\eta$')
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(0,1.1)
    if lt != overview:
        ax.text(-1.5, 1.1, "layer triplet %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('physics_eta_%s.pdf'%str(int(lt)))

#******************************
# plot over eta

for data, lt in zip(pts, ttbar['layerTriplet']):
    fig, ax = plt.subplots()
    ax.plot(data['bin'], data['valid'] / (data['valid'] + data['missed']), 'go', label='efficiency')
    ax.plot(data['bin'], data['fake'] / (data['valid'] + data['fake'] + data['clones']), 'ro', label='fake rate')
    ax.plot(data['bin'], data['clones'] / (data['valid'] + data['fake'] + data['clones']), 'yo', label='clone rate')
    
    ax.set_title(r'$t\bar{t}$ simulated event studies over $p_T$')
    ax.set_xlabel(r'$p_t$ in [$\frac{GeV}{c}$]')
    ax.set_xlim(0,50)
    ax.set_ylim(0,1.1)
    if lt != overview:
        ax.text(0, 1.1, "layer triplet %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('physics_pt_%s.pdf'%str(int(lt)))

plt.show()
