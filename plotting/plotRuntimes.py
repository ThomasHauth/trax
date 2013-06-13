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
