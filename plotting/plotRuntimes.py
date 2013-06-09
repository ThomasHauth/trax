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

import xkcdify as xkcd

 # multiple events concurrently
events_cpu = np.genfromtxt("runtime.multipleEvents.test.cpu.csv", delimiter = " ", names=True)
events_gpu = np.genfromtxt("runtime.multipleEvents.test.gpu.csv", delimiter = " ", names=True)

# varying work-group sizes
threads_gpu = np.genfromtxt("runtime.threads.test.gpu.csv", delimiter = " ", names=True)

# varying tracks per event
tracks_gpu = np.genfromtxt("runtime.tracks.test.gpu.csv", delimiter = " ", names=True)
tracks_gpu.sort(order='tracks')

# cmssw data
cmssw_org = np.genfromtxt("cmsswTiming.org.csv", delimiter = " ", names=True)
cmssw_org.sort(order='tracks')

cmssw_kd = np.genfromtxt("cmsswTiming.kd.csv", delimiter = " ", names=True)
cmssw_kd.sort(order='tracks')

cmssw_6 = np.genfromtxt("cmsswTiming.6.csv", delimiter = " ", names=True)
cmssw_6.sort(order='tracks')

###################################################
# hardware plots - time and throughput

prefixSum_gpu = np.genfromtxt("data/runtime/runtime.prefixSum.gpu.csv", delimiter = " ", names=True)
prefixSum_cpu = np.genfromtxt("data/runtime/runtime.prefixSum.cpu.csv", delimiter = " ", names=True)

#prefixSum_gpu2 = np.genfromtxt("data/runtime/runtime.prefixSum.gpu2.csv", delimiter = " ", names=True)
#prefixSum_cpu2 = np.genfromtxt("data/runtime/runtime.prefixSum.cpu2.csv", delimiter = " ", names=True)

fig = plt.figure()

plt.title(r'Prefix Sum - Execution Time')

ax1 = fig.add_subplot(111)
#ax1.plot(prefixSum_cpu2['wgSize'], prefixSum_cpu2['time'], 'bx--', label='Core i3')
ax1.plot(prefixSum_cpu['wgSize'], prefixSum_cpu['time'], 'bo-', label='Core i7')
#ax1.plot(prefixSum_gpu2['wgSize'], prefixSum_gpu2['time'], 'gx--', label='GTX 560')
ax1.plot(prefixSum_gpu['wgSize'], prefixSum_gpu['time'], 'go-', label='GTX 660')
ax1.set_xlabel('work-group size')
ax1.set_ylabel(r'time [ms]')
ax1.set_xscale('log', basex=2)

ax1.legend(loc=1)

plt.savefig('output/runtime/runtime_prefixSum.pdf')

###################################################
# hardware plots - speedup

fig = plt.figure()

plt.title(r'Prefix Sum - Speedup')

ax1 = fig.add_subplot(111)
#ax1.plot(prefixSum_cpu2['wgSize'], prefixSum_cpu2[prefixSum_cpu2['wgSize'] == 1]['time'] / prefixSum_cpu2['time'], 'bx--', label='Core i3')
#ax1.plot(prefixSum_gpu2['wgSize'], prefixSum_gpu2[prefixSum_gpu2['wgSize'] == 1]['time'] / prefixSum_gpu2['time'], 'gx--', label='GTX 560')
ax1.plot(prefixSum_gpu['wgSize'], prefixSum_gpu[prefixSum_gpu['wgSize'] == 1]['time'] / prefixSum_gpu['time'], 'go-', label='GTX 660 - relative')
ax1.plot(prefixSum_gpu['wgSize'], prefixSum_cpu[prefixSum_cpu['wgSize'] == 1]['time'] / prefixSum_gpu['time'], 'gx--', label='GTX 660 - absolute')
ax1.plot(prefixSum_gpu['wgSize'], prefixSum_cpu['time'] / prefixSum_gpu['time'], 'kD:', label='GTX 660 over Core i7')
ax1.plot(prefixSum_cpu['wgSize'], prefixSum_cpu[prefixSum_cpu['wgSize'] == 1]['time'] / prefixSum_cpu['time'], 'bo-', label='Core i7')
ax1.set_xlabel('work-group size')
ax1.set_ylabel(r'time [ms]')
ax1.set_xscale('log', basex=2)

ax1.legend(loc=2)

plt.savefig('output/runtime/speedup_prefixSum.pdf')

###################################################
# time for n concurrent events
plt.figure()
plt.errorbar(events_cpu['events'], events_cpu['totalWalltime'], yerr=events_cpu['totalWalltimeVar'], fmt='bo-', label='wall time CPU')
plt.errorbar(events_cpu['events'], events_cpu['totalKernel'], yerr=events_cpu['totalKernelVar'], fmt='bo--', label='kernel time CPU')
plt.errorbar(events_gpu['events'], events_gpu['totalWalltime'], yerr=events_gpu['totalWalltimeVar'], fmt='go-', label='wall time GPU')
plt.errorbar(events_gpu['events'], events_gpu['totalKernel'], yerr=events_gpu['totalKernelVar'], fmt='go--', label='kernel time GPU')

plt.title(r'Timing for processing $n$ events concurrently')
plt.xlabel('events')
plt.ylabel('time [ms]')
plt.xlim(np.min(events_gpu['events']), np.max(events_gpu['events']))

plt.legend(ncol=2, loc=2)

plt.savefig('runtime_concurrent_events.pdf')


###################################################
# time per event for n concurrent events
plt.figure()
plt.errorbar(events_cpu['events'], events_cpu['totalWalltime'] / events_cpu['events'], yerr=events_cpu['totalWalltimeVar'] / events_cpu['events'], fmt='bo-', label='wall time CPU')
plt.errorbar(events_cpu['events'], events_cpu['totalKernel'] / events_cpu['events'], yerr=events_cpu['totalKernelVar'] / events_cpu['events'], fmt='bo--', label='kernel time CPU')
plt.errorbar(events_gpu['events'], events_gpu['totalWalltime'] / events_gpu['events'], yerr=events_gpu['totalWalltimeVar'] / events_gpu['events'], fmt='go-', label='wall time GPU')
plt.errorbar(events_gpu['events'], events_gpu['totalKernel'] / events_gpu['events'], yerr=events_gpu['totalKernelVar'] / events_gpu['events'], fmt='go--', label='kernel time GPU')

plt.title('Processing time per event')
plt.xlabel('events')
plt.ylabel('time / event [ms]')
plt.xlim(np.min(events_gpu['events']), np.max(events_gpu['events']))
plt.ylim(0,50)

plt.legend(ncol=2)

plt.savefig('runtime_per_event.pdf')

###################################################
# runtime contributions (events)
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))
total = events_gpu['totalKernel'] + events_gpu['readTime'] + events_gpu['writeTime']

colors = ['#003300', '#006600', 
          'r', 
          'c', 
          '#3366CC', '#000099',
          'm']
labels = ['IO - read', 'IO - write',
          'Build grid', 
          'Gen pairs', 
          'Predict triplets - count', 'Predict triplets - store', 
          'Filter triplets']

legendHandles = []
for c in colors:
    legendHandles.append(plt.Rectangle((0, 0), 1, 1, fc=c))

ax.stackplot(events_gpu['events'], 
              events_gpu['readTime'] / total, events_gpu['writeTime'] / total,
              events_gpu['buildGridKernel'] / total, 
              events_gpu['pairGenKernel'] / total, 
              (events_gpu['tripletPredictCount'] + events_gpu['tripletPredictScan']) / total, events_gpu['tripletPredictStore'] / total,
              events_gpu['tripletFilterKernel'] / total,
              colors = colors)

ax.set_xlabel('events')
ax.set_ylabel('runtime share [%]')
ax.set_title('Composition of event processing runtime')
ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
ax.set_xlim(np.min(events_gpu['events']), np.max(events_gpu['events']))
ax.set_ylim(0,1)

plt.savefig('runtime_contrib_events.pdf')

###################################################
# runtime contributions (tracks)

fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))
total = tracks_gpu['totalKernel'] + tracks_gpu['readTime'] + tracks_gpu['writeTime']

colors = ['#003300', '#006600', 
          'r', 
          'c', 
          '#3366CC', '#000099',
          'm']
labels = ['IO - read', 'IO - write',
          'Build grid', 
          'Gen pairs', 
          'Predict triplets - count', 'Predict triplets - store', 
          'Filter triplets']

legendHandles = []
for c in colors:
    legendHandles.append(plt.Rectangle((0, 0), 1, 1, fc=c))

ax.stackplot(tracks_gpu['tracks'], 
              tracks_gpu['readTime'] / total, tracks_gpu['writeTime'] / total,
              tracks_gpu['buildGridKernel'] / total, 
              tracks_gpu['pairGenKernel'] / total, 
              (tracks_gpu['tripletPredictCount'] + tracks_gpu['tripletPredictScan']) / total, tracks_gpu['tripletPredictStore'] / total,
              tracks_gpu['tripletFilterKernel'] / total,
              colors = colors)

ax.set_xlabel('tracks')
ax.set_ylabel('runtime share [%]')
ax.set_title('Composition of event processing runtime')
ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
ax.set_xlim(np.min(tracks_gpu['tracks']), np.max(tracks_gpu['tracks']))
ax.set_ylim(0,1)

plt.savefig('runtime_contrib_tracks.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(threads_gpu['threads'], threads_gpu['totalWalltime'], yerr=threads_gpu['totalWalltimeVar'], fmt='go-', label='wall time GPU')
plt.errorbar(threads_gpu['threads'], threads_gpu['totalKernel'], yerr=threads_gpu['totalKernelVar'], fmt='go--', label='kernel time GPU')

plt.title('Processing time with work-group size')
plt.xlabel('threads')
plt.ylabel(r'time [ms]')
plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
plt.yscale("log")

plt.legend()

plt.savefig('runtime_wg_size.pdf')

###################################################
# speedup for varying work-group sizes
plt.figure()

walltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalWalltime']
kerneltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalKernel']

plt.plot(threads_gpu['threads'], threads_gpu['threads'], 'b:', label='ideal')
plt.plot(threads_gpu['threads'], walltimeGPU1/ threads_gpu['totalWalltime'], 'go-', label='speedup wall time GPU')
plt.plot(threads_gpu['threads'], kerneltimeGPU1 / threads_gpu['totalKernel'], 'go--', label='speedup kernel time GPU')

plt.title('Speedup for varying work-group sizes')
plt.xlabel('threads')
plt.ylabel(r'speedup')
plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
#plt.xscale('log')

plt.legend()

plt.savefig('runtime_speedup.pdf')

###################################################
# efficiency for varying work-group sizes
plt.figure()

walltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalWalltime']
kerneltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalKernel']

kernelScan1 = threads_gpu[threads_gpu['threads'] == 1]['buildGridScan']

plt.plot(threads_gpu['threads'], (walltimeGPU1/ threads_gpu['totalWalltime']) / threads_gpu['threads'], 'go-', label='efficiency wall time GPU')
plt.plot(threads_gpu['threads'], (kerneltimeGPU1 / threads_gpu['totalKernel'])/ threads_gpu['threads'], 'go--', label='efficiency kernel time GPU')

plt.title('Efficiency for varying work-group sizes')
plt.xlabel('threads')
plt.ylabel(r'efficiency')
plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
#plt.xscale('log')

plt.legend()

plt.savefig('runtime_eff.pdf')

###################################################
# processing time with tracks per event
plt.figure()

plt.errorbar(tracks_gpu['tracks'] / tracks_gpu['events'], tracks_gpu['totalWalltime'] / tracks_gpu['events'], yerr=tracks_gpu['totalWalltimeVar'] / tracks_gpu['events'],fmt='go-', label='wall time GPU')
plt.errorbar(tracks_gpu['tracks'] / tracks_gpu['events'], tracks_gpu['totalKernel'] / tracks_gpu['events'], yerr=tracks_gpu['totalKernelVar'] / tracks_gpu['events'],fmt='gx--', label='kernel time GPU')

plt.errorbar(cmssw_org['validTracks'], cmssw_org['time'], yerr=cmssw_org['timeVar'] ,fmt='yo-', label='CMSSW 5.2.4')
#plt.errorbar(cmssw_kd['validTracks'], cmssw_kd['time'], yerr=cmssw_kd['timeVar'] ,fmt='ro-', label=r'CMSSW $k$-d tree')
plt.errorbar(cmssw_kd['validTracks'], cmssw_6['time'], yerr=cmssw_6['timeVar'] ,fmt='ro-', label=r'CMSSW 6.0.0')

plt.title(r'Processing time for $n$ tracks per event')
plt.xlabel('tracks')
plt.ylabel(r'time / event [ms]')
#plt.xlim(np.min(tracks_gpu['tracks']), np.max(tracks_gpu['tracks']))
plt.xlim(xmin=1)
plt.loglog()
plt.ylim(ymin=0)

plt.legend(loc=2)

plt.savefig('runtime_tracks_cmssw.pdf')

###################################################
# grid building time with hits per event
plt.figure()

#plt.errorbar(tracks_gpu['hits'], tracks_gpu['buildGridWalltime'], yerr=tracks_gpu['buildGridWalltimeVar'] / tracks_gpu['hits'] ,fmt='go-', label='wall time GPU')
plt.errorbar(tracks_gpu['hits'], tracks_gpu['buildGridKernel'] / tracks_gpu['hits'], yerr=tracks_gpu['buildGridKernelVar'] / tracks_gpu['hits'] ,fmt='go--', label='kernel time GPU')

plt.title(r'Grid building for $n$ hits')
plt.xlabel('hits')
plt.ylabel(r'time / hits [ms]')
#plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
plt.yscale('log')

plt.legend()

plt.savefig('runtime_grid.pdf')

###################################################
# data transferred (tracks)
fig = plt.figure()

plt.title(r'Data transfer volumne')

ax1 = fig.add_subplot(111)
t = ax1.plot(tracks_gpu['tracks'], (tracks_gpu['writeBytes'] / 10**6 ), 'go-', label='transferred bytes')
ax1.set_xlabel('tracks')
ax1.set_ylabel(r'[MB]')

ax2 = ax1.twinx()
ax2.set_ylabel('bandwith [GB/s]')
b = ax2.plot(tracks_gpu['tracks'], tracks_gpu['writeBytes'] / (10**6 * tracks_gpu['writeTime']), 'bo--', label='bandwith')

ax1.legend(loc=2)
ax2.legend(loc=1)

plt.savefig('runtime_transfer_tracks.pdf')

###################################################
# data transferred (events)
fig = plt.figure()

plt.title(r'Data transfer volumne')

ax1 = fig.add_subplot(111)
t = ax1.plot(events_gpu['events'], (events_gpu['writeBytes'] / 10**6 ), 'go-', label='transferred bytes')
ax1.set_xlabel('events')
ax1.set_ylabel(r'[MB]')

ax2 = ax1.twinx()
ax2.set_ylabel('bandwith [GB/s]')
b = ax2.plot(events_gpu['events'], events_gpu['writeBytes'] / (10**6 * events_gpu['writeTime']), 'bo--', label='bandwith')

ax1.legend(loc=2)
ax2.legend(loc=1)

plt.savefig('runtime_transfer_events.pdf')

plt.show()
