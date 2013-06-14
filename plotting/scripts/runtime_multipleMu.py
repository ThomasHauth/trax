import numpy as np

import latexHeader

import matplotlib.pyplot as plt

def filter(data, events):
    return data[data['events'] == events]

local_gpu = np.genfromtxt("../data/runtime/runtime.multipleMu.local.conf.gpu.csv", delimiter = " ", names=True)
local_gpu.sort(order='tracks')

halfLocal_gpu = np.genfromtxt("../data/runtime/runtime.multipleMu.halfLocal.conf.gpu.csv", delimiter = " ", names=True)
halfLocal_gpu.sort(order='tracks')

noLocal_gpu = np.genfromtxt("../data/runtime/runtime.multipleMu.noLocal.conf.gpu.csv", delimiter = " ", names=True)
noLocal_gpu.sort(order='tracks')

superFine_gpu = np.genfromtxt("../data/runtime/runtime.multipleMu.superFine.conf.gpu.csv", delimiter = " ", names=True)
superFine_gpu.sort(order='tracks')

local_cpu = np.genfromtxt("../data/runtime/runtime.multipleMu.local.conf.cpu.csv", delimiter = " ", names=True)
local_cpu.sort(order='tracks')

halfLocal_cpu = np.genfromtxt("../data/runtime/runtime.multipleMu.halfLocal.conf.cpu.csv", delimiter = " ", names=True)
halfLocal_cpu.sort(order='tracks')

noLocal_cpu = np.genfromtxt("../data/runtime/runtime.multipleMu.noLocal.conf.cpu.csv", delimiter = " ", names=True)
noLocal_cpu.sort(order='tracks')

superFine_cpu = np.genfromtxt("../data/runtime/runtime.multipleMu.superFine.conf.cpu.csv", delimiter = " ", names=True)
superFine_cpu.sort(order='tracks')

cmssw_5 = np.genfromtxt("../data/runtime/runtime.multipleMu.cmssw5.csv", delimiter = " ", names=True)
cmssw_5.sort(order='tracks')

cmssw_6 = np.genfromtxt("../data/runtime/runtime.multipleMu.cmssw6.csv", delimiter = " ", names=True)
cmssw_6.sort(order='tracks')

data = [(local_gpu, local_cpu, 'coarse grid'), (halfLocal_gpu, halfLocal_cpu, 'medium grid'), (noLocal_gpu, noLocal_cpu, 'fine grid'), (superFine_gpu, superFine_cpu, 'super fine grid')]

###############################################
## individual comparison
for item in data:

    data_gpu = item[0]
    data_cpu = item[1]
    
    fig = plt.figure()
    top = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
    bottom = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=top)

    #cms5 = top.errorbar(cmssw_5['validTracks'], cmssw_5['time'], yerr=cmssw_5['timeVar'] ,fmt='yo-', label='CMSSW 5.2.4')
    cms = top.errorbar(cmssw_6['validTracks'], cmssw_6['time'], yerr=cmssw_6['timeVar'] ,fmt='ro-', label=r'CMSSW 6.0.0')

    gw = top.errorbar(data_gpu['tracks'] / data_gpu['events'], data_gpu['totalWalltime'] / data_gpu['events'], yerr=data_gpu['totalWalltimeVar'] / data_gpu['events'],fmt='gx--', label='wall time GPU')
    gk = top.errorbar(data_gpu['tracks'] / data_gpu['events'], data_gpu['totalKernel'] / data_gpu['events'], yerr=data_gpu['totalKernelVar'] / data_gpu['events'],fmt='go-', label='kernel time GPU')
    
    cw = top.errorbar(data_cpu['tracks'] / data_cpu['events'], data_cpu['totalWalltime'] / data_cpu['events'], yerr=data_cpu['totalWalltimeVar'] / data_cpu['events'],fmt='bx--', label='wall time CPU')
    ck = top.errorbar(data_cpu['tracks'] / data_cpu['events'], data_cpu['totalKernel'] / data_cpu['events'], yerr=data_cpu['totalKernelVar'] / data_cpu['events'],fmt='bo-', label='kernel time CPU')
    
    top.set_title(r'Processing Time over Tracks')
    
    top.set_ylabel(r'time / event [ms]')
    #plt.xlim(np.min(tracks_gpu['tracks']), np.max(tracks_gpu['tracks']))
    
    top.set_xlim(xmin=1)
    top.set_xscale('log')
    top.set_yscale('log')
    top.set_ylim(0, 2*10**5)
    
    l = [gw,gk,cw,ck,cms]
    
    a = top.legend(l, [x.get_label() for x in l], loc=2, ncol=3, mode='expand')
    fig.text(0.11, 0.93, "%s"%item[2], va='bottom', ha='left')
    
    bottom.set_xlabel('tracks / event')
    bottom.set_xscale('log')
    bottom.set_yscale('log', basey=4)
    bottom.set_ylabel('ratio')
    bottom.get_yaxis().get_major_formatter().base(2)
    
    bottom.plot(cmssw_6['validTracks'],  cmssw_6['time'] / cmssw_6['time'], 'ro-')
    
    bottom.plot(data_gpu['tracks'] / data_gpu['events'],  cmssw_6['time'] / (data_gpu['totalKernel'] / data_gpu['events']), 'go-')
    bottom.plot(data_cpu['tracks'] / data_cpu['events'],  cmssw_6['time'] / (data_cpu['totalKernel'] / data_cpu['events']), 'bo-')
    #bottom.plot(cmssw_5['validTracks'],  cmssw_5['time'] / cmssw_5['time'], 'yo-')
    
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    plt.savefig('../output/runtime/runtime_multipleMu_%s.pdf'%item[2].replace (" ", "_"))
    
###################################################
# kernel time for everything
fig = plt.figure()
top = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
bottom = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=top)

g1 = top.errorbar(local_gpu['tracks'] / 30, local_gpu['totalKernel'] / local_gpu['events'], yerr=local_gpu['totalKernelVar']  / local_gpu['events'], fmt='go-', label=r'coarse grid - GPU')
g2 = top.errorbar(halfLocal_gpu['tracks'] / 30, halfLocal_gpu['totalKernel'] / halfLocal_gpu['events'], yerr=local_gpu['totalKernelVar']  / local_gpu['events'], fmt='gx--', label=r'medium grid - GPU')
g3 = top.errorbar(noLocal_gpu['tracks'] / 30, noLocal_gpu['totalKernel']  / noLocal_gpu['events'], yerr=noLocal_gpu['totalKernelVar']  / noLocal_gpu['events'], fmt='gD:', label=r'fine grid - GPU')

#g4 = top.errorbar(superFine_gpu['tracks'] / 30, superFine_gpu['totalKernel']  / superFine_gpu['events'], yerr=superFine_gpu['totalKernelVar']  / superFine_gpu['events'], fmt='rD:', label=r'super fine grid - GPU')

c1 = top.errorbar(local_cpu['tracks'] / 30, local_cpu['totalKernel'] / local_cpu['events'], yerr=local_cpu['totalKernelVar']  / local_cpu['events'], fmt='bo-', label=r'coarse grid - CPU')
c2 = top.errorbar(halfLocal_cpu['tracks'] / 30, halfLocal_cpu['totalKernel'] / halfLocal_cpu['events'], yerr=local_cpu['totalKernelVar']  / local_cpu['events'], fmt='bx--', label=r'medium grid - CPU')
c3 = top.errorbar(noLocal_cpu['tracks'] / 30, noLocal_cpu['totalKernel']  / noLocal_cpu['events'], yerr=noLocal_cpu['totalKernelVar']  / noLocal_cpu['events'], fmt='bD:', label=r'fine grid - CPU')

#c4 = top.errorbar(superFine_cpu['tracks'] / 30, superFine_cpu['totalKernel']  / superFine_cpu['events'], yerr=superFine_cpu['totalKernelVar']  / superFine_cpu['events'], fmt='rD:', label=r'super fine grid - CPU')

one = [1 for i in range(len(halfLocal_gpu['tracks']))]
bottom.plot(halfLocal_gpu['tracks'] / 30, one, 'ko-')

#bottom.plot(local_gpu['tracks'] / 30, local_gpu['totalKernel'] / local_gpu['totalKernel'], 'go-', label=r'coarse grid - GPU')
bottom.plot(halfLocal_gpu['tracks'] / 30, local_gpu['totalKernel'] / halfLocal_gpu['totalKernel'], 'gx--', label=r'medium grid - GPU')
bottom.plot(noLocal_gpu['tracks'] / 30, local_gpu['totalKernel'] / noLocal_gpu['totalKernel'], 'gD:', label=r'fine grid - GPU')

#bottom.plot(local_cpu['tracks'] / 30, local_cpu['totalKernel'] / local_cpu['totalKernel'], 'bo-', label=r'coarse grid - CPU')
bottom.plot(halfLocal_cpu['tracks'] / 30, local_cpu['totalKernel'] / halfLocal_cpu['totalKernel'], 'bx--', label=r'medium grid - CPU')
bottom.plot(noLocal_cpu['tracks'] / 30, local_cpu['totalKernel'] / noLocal_cpu['totalKernel'], 'bD:', label=r'fine grid - CPU')


top.set_title('Processing Time over Tracks')
bottom.set_xlabel('tracks / event')
top.set_ylabel(r'time [ms]')
bottom.set_ylabel(r'ratio')
#plt.ylim(ymax=20)
bottom.set_xscale('log', basex=10)
top.set_xscale('log', basex=10)
top.set_yscale("log", basey=10)

#l1 = [g1,g2,g3]
a = top.legend(loc=2, ncol=2, columnspacing=1)

#l2 = [c1,c2,c3]
#top.legend(l2, [x.get_label() for x in l2], loc=9)
#top.add_artist(a)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig('../output/runtime/runtime_multipleMu.pdf')
        