import numpy as np

import latexHeader

import matplotlib.pyplot as plt

###################################################
# hardware plots - time and throughput

prefixSum_gpu = np.genfromtxt("../data/runtime/runtime.prefixSum.gpu.csv", delimiter = " ", names=True)
prefixSum_cpu = np.genfromtxt("../data/runtime/runtime.prefixSum.cpu.csv", delimiter = " ", names=True)

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

plt.savefig('../output/runtime/runtime_prefixSum.pdf')

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
ax1.set_yscale('log', basey=2)

ax1.legend(loc=2)

plt.savefig('../output/runtime/speedup_prefixSum.pdf')