import csv
import numpy as np
import matplotlib.pyplot as plt

 # multiple events concurrently
events_cpu = np.genfromtxt("runtime.multipleEvents.test.cpu.csv", delimiter = " ", names=True)
events_gpu = np.genfromtxt("runtime.multipleEvents.test.gpu.csv", delimiter = " ", names=True)

# varying work-group sizes
threads_gpu = np.genfromtxt("runtime.threads.test.gpu.csv", delimiter = " ", names=True)


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

###################################################
# runtime contributions
plt.figure()
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

plt.stackplot(events_gpu['events'], 
              events_gpu['readTime'] / total, events_gpu['writeTime'] / total,
              events_gpu['buildGridKernel'] / total, 
              events_gpu['pairGenKernel'] / total, 
              (events_gpu['tripletPredictCount'] + events_gpu['tripletPredictScan']) / total, events_gpu['tripletPredictStore'] / total,
              events_gpu['tripletFilterKernel'] / total,
              colors = colors)

plt.xlabel('events')
plt.ylabel('runtime share [%]')
plt.title('Composition of event processing runtime')
plt.figlegend(legendHandles[::-1], labels[::-1], 7)
plt.xlim(np.min(events_gpu['events']), np.max(events_gpu['events']))
plt.ylim(0,1)

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

###################################################
# speedup for varying work-group sizes
plt.figure()

walltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalWalltime']
kerneltimeGPU1 = threads_gpu[threads_gpu['threads'] == 1]['totalKernel']

print walltimeGPU1, kerneltimeGPU1

plt.plot(threads_gpu['threads'], threads_gpu['threads'], ':', label='ideal')
plt.plot(threads_gpu['threads'], walltimeGPU1/ threads_gpu['totalWalltime'], 'go-', label='speedup wall time GPU')
plt.plot(threads_gpu['threads'], kerneltimeGPU1 / threads_gpu['totalKernel'], 'go--', label='speedup kernel time GPU')

plt.title('Speedup for varying work-group sizes')
plt.xlabel('threads')
plt.ylabel(r'speedup [ms]')
plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
plt.xscale('log')

plt.legend()

plt.show()
