import numpy as np

import latexHeader

import matplotlib.pyplot as plt
import math

def binomialError(data):
    return [1/math.sqrt(x) if x > 0 else 0 for x in data]

def div(a,b):
    return [x / y if y > 0 else 0 for (x,y) in zip(a,b)]

# tt bar physics
ttbar = np.genfromtxt("../data/physics/physics.ttBar.csv", delimiter = " ", names=True)

# load eta and pt histograms
etas = []
pts = []

for lt in ttbar['layerTriplet']:
    etas.append(np.genfromtxt("../data/physics/ttbar/eta_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    pts.append(np.genfromtxt("../data/physics/ttbar/pt_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    
#define layer triplets
layerTriplets = ["", "1-2-3", "2-3-4", "3-4-5", "4-5-8", "", "combined"]
overview = 5

#******************************
# overview plot

plt.figure()
plt.errorbar(ttbar['layerTriplet']+1, ttbar['eff'], yerr=ttbar['effVar'], fmt='go-', label='efficiency')
plt.errorbar(ttbar['layerTriplet']+1, ttbar['fr'], yerr=ttbar['frVar'], fmt='ro-', label='fake rate')
plt.errorbar(ttbar['layerTriplet']+1, ttbar['cr'], yerr=ttbar['crVar'], fmt='yo-', label='clone rate')

plt.title(r'$t\bar{t}$ Simulated Event Studies')
plt.xlabel("layer triplet")
plt.xlim(0, overview+2)
plt.gca().set_xticklabels(layerTriplets)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)

plt.savefig('../output/physics/physics_ttbar_overview.pdf')

#******************************
# plot over eta

for data, lt in zip(etas, ttbar['layerTriplet']):
    fig, ax = plt.subplots()
    
    ax.errorbar(data['bin'], div(data['valid'], (data['valid'] + data['missed'])), yerr=binomialError(data['valid']), fmt='go', label='efficiency')
    ax.errorbar(data['bin'], div(data['fake'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['fake']), fmt='ro', label='fake rate')
    ax.errorbar(data['bin'], div(data['clones'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['clones']), fmt='yo', label='clone rate')
    
    ax.set_title(r'$t\bar{t}$ Simulated Event Studies over $\eta$')
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(0,1.1)
    if lt != overview:
        ax.text(-1.5, 1.1, "layer triplet %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('../output/physics/physics_ttbar_eta_%s.pdf'%str(int(lt)))

#******************************
# plot over pt

maxPt = 50

for dat, lt in zip(pts, ttbar['layerTriplet']):
    fig, ax = plt.subplots()
    
    data = dat[dat['bin'] <= maxPt]
    
    ax.errorbar(data['bin'], div(data['valid'], (data['valid'] + data['missed'])), yerr=binomialError(data['valid']), fmt='go', label='efficiency')
    ax.errorbar(data['bin'], div(data['fake'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['fake']), fmt='ro', label='fake rate')
    ax.errorbar(data['bin'], div(data['clones'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['clones']), fmt='yo', label='clone rate')
    
    ax.set_title(r'$t\bar{t}$ Simulated Event Studies over $p_T$')
    ax.set_xlabel(r'$p_t$ in [$GeV/c$]')
    ax.set_xlim(0,maxPt)
    ax.set_ylim(0,1.1)
    if lt != overview:
        ax.text(0, 1.1, "layer triplet %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('../output/physics/physics_ttbar_pt_%s.pdf'%str(int(lt)))