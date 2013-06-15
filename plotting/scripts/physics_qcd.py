import numpy as np

#import latexHeader

import matplotlib.pyplot as plt
import math

def binomialError(data):
    return [1/math.sqrt(x) if x > 0 else 0 for x in data]

def div(a,b):
    return [x / y if y > 0 else 0 for (x,y) in zip(a,b)]


minPt = 1
maxPt = 100

def rebin(data):
    rebinned = np.zeros(0, dtype=data.dtype)
    print data
    
    pt = minPt    
    while pt <= maxPt:
        binWidth = pt / 10.0
        print pt, "-", pt + binWidth
        dat = data[(data['bin'] >= pt) & (data['bin'] < pt + binWidth)]
        n = np.array([(pt + binWidth/2.0, dat['valid'].sum(), dat['fake'].sum(), dat['clones'].sum(), dat['missed'].sum())], dtype=data.dtype)
        print n
        rebinned = np.append(rebinned, n)
        pt += binWidth
    
#    for i in range(minPt, maxPt+1):
#        bins = math.ceil(10.0/i)
#        for j in range (0, int(bins)):
#            print (i + j / bins), "-", (i + (j+1)/bins)
#            dat = data[(data['bin'] >= (i + j / bins)) & (data['bin'] < (i + (j+1)/bins))]
#            #print dat
#            n = np.array([(i + j / bins, dat['valid'].sum(), dat['fake'].sum(), dat['clones'].sum(), dat['missed'].sum())], dtype=data.dtype)
#            print n
#            rebinned = np.append(rebinned, n)
            
    print rebinned
    return rebinned

# tt bar physics
qcd = np.genfromtxt("../data/physics/physics.qcd.csv", delimiter = " ", names=True)

# load eta and pt histograms
etas = []
pts = []

for lt in qcd['layerTriplet']:
    etas.append(np.genfromtxt("../data/physics/qcd/eta_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    pts.append(np.genfromtxt("../data/physics/qcd/pt_%s.csv"%str(int(lt)), delimiter = " ", names=True))
    
#define layer triplets
layerTriplets = ["", "1-2-3", "2-3-4", "3-4-5", "4-5-8", "", "combined"]
overview = 5

#******************************
# overview plot

plt.figure()
plt.errorbar(qcd['layerTriplet']+1, qcd['eff'], yerr=qcd['effVar'], fmt='go-', label='efficiency')
plt.errorbar(qcd['layerTriplet']+1, qcd['fr'], yerr=qcd['frVar'], fmt='ro-', label='fake rate')
plt.errorbar(qcd['layerTriplet']+1, qcd['cr'], yerr=qcd['crVar'], fmt='yo-', label='clone rate')

plt.title(r'QCD Simulated Event Studies')
plt.xlabel("layer triplet")
plt.xlim(0, overview+2)
plt.gca().set_xticklabels(layerTriplets)

plt.ylim(-0.1, 1.1)
plt.axhline(1, linestyle='--', color='k', linewidth=0.5)
plt.axhline(0, linestyle='--', color='k', linewidth=0.5)

plt.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)

plt.savefig('../output/physics/physics_qcd_overview.pdf')

#******************************
# plot over eta

for data, lt in zip(etas, qcd['layerTriplet']):
    fig, ax = plt.subplots()
    
    ax.errorbar(data['bin'], div(data['valid'], (data['valid'] + data['missed'])), yerr=binomialError(data['valid']), fmt='go', label='efficiency')
    ax.errorbar(data['bin'], div(data['fake'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['fake']), fmt='ro', label='fake rate')
    ax.errorbar(data['bin'], div(data['clones'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['clones']), fmt='yo', label='clone rate')
    
    ax.set_title(r'QCD Simulated Event Studies over $\eta$')
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-0.1,1.1)
    
    ax.axhline(1, linestyle='--', color='k', linewidth=0.5)
    ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
    
    if lt != overview:
        fig.text(0.12, 0.9, "layers %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('../output/physics/physics_qcd_eta_%s.pdf'%str(int(lt)))

#******************************
# plot over pt

for dat, lt in zip(pts, qcd['layerTriplet']):
    fig, ax = plt.subplots()
    
    data = rebin(dat)
    
    ax.errorbar(data['bin'], div(data['valid'], (data['valid'] + data['missed'])), yerr=binomialError(data['valid']), fmt='go', label='efficiency')
    ax.errorbar(data['bin'], div(data['fake'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['fake']), fmt='ro', label='fake rate')
    ax.errorbar(data['bin'], div(data['clones'], (data['valid'] + data['fake'] + data['clones'])), yerr=binomialError(data['clones']), fmt='yo', label='clone rate')
    
    ax.set_title(r'QCD Simulated Event Studies over $p_T$')
    ax.set_xlabel(r'$p_t$ in [$GeV/c$]')
    ax.set_xlim(minPt,maxPt)
    #ax.set_xscale('symlog', linthreshx=10, subsx=[0,1,2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xscale('log', nonposx='clip')
    ax.set_ylim(-.1,1.1)
    ax.axhline(1, linestyle='--', color='k', linewidth=0.5)
    ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
    
    if lt != overview:
        fig.text(0.12, 0.9, "layers %s"%layerTriplets[int(lt)+1], va='bottom', ha='left')
    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=fig.transFigure)
    plt.savefig('../output/physics/physics_qcd_pt_%s.pdf'%str(int(lt)))
    