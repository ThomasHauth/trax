import numpy as np

import latexHeader

import matplotlib.pyplot as plt
import math

# cmssw data
cmssw = np.genfromtxt("../data/physics/physics.multipleMu.cmssw.csv", delimiter = " ", names=True)
cmssw.sort(order='tracks')

#cmssw physics comparison
ocl = np.genfromtxt("../data/physics/physics.multipleMu.csv", delimiter = " ", names=True)
ocl.sort(order='tracks')

#******************************
#plot cmssw over tracks
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))

ax.plot(ocl['tracks'], ocl['eff'], 'go-', label=r'OCL efficiency')
ax.plot(ocl['tracks'], ocl['fr'], 'ro-', label=r'OCL fake rate')
ax.plot(ocl['tracks'], ocl['cr'], 'yo-', label=r'OCL clone rate')

ax.plot(cmssw['tracks'], cmssw['eff'], 'gx--', label='CMSSW efficiency')
ax.plot(cmssw['tracks'], cmssw['fr'], 'rx--', label='CMSSW fake rate')
ax.plot(cmssw['tracks'], cmssw['cr'], 'yx--', label='CMSSW clone rate')

ax.set_title('Physics Performance over Tracks')
ax.set_xlabel('tracks')
ax.set_xscale('log')
ax.set_ylim(0, 1.1)
ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure, loc=5)

plt.savefig('../output/physics/physics_multipleMu.pdf')