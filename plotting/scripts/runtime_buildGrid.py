import numpy as np

import latexHeader

import matplotlib.pyplot as plt

def filter(data, events):
    return data[data['events'] == events]

local_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.tracks.test.gpu.csv", delimiter = " ", names=True)
halfLocal_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.halfLocal.conf.tracks.test.gpu.csv", delimiter = " ", names=True)
noLocal_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.noLocal.conf.tracks.test.gpu.csv", delimiter = " ", names=True)

local_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.tracks.test.cpu.csv", delimiter = " ", names=True)
halfLocal_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.halfLocal.conf.tracks.test.cpu.csv", delimiter = " ", names=True)
noLocal_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.noLocal.conf.tracks.test.cpu.csv", delimiter = " ", names=True)

###############################
######## GPU

###################################################
# kernel time for grid building
fig = plt.figure()
top = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
bottom = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=top)

g1 = top.errorbar(filter(noLocal_gpu, 30)['hits'] / 30, filter(noLocal_gpu, 30)['buildGridKernel']  / filter(noLocal_gpu, 30)['hits'], yerr=filter(noLocal_gpu, 30)['buildGridKernelVar']  / filter(noLocal_gpu, 30)['hits'], fmt='gD:', label=r'coarse grid - GPU')
g2 = top.errorbar(filter(halfLocal_gpu, 30)['hits'] / 30, filter(halfLocal_gpu, 30)['buildGridKernel']  / filter(halfLocal_gpu, 30)['hits'], yerr=filter(halfLocal_gpu, 30)['buildGridKernelVar']  / filter(halfLocal_gpu, 30)['hits'], fmt='gx--', label=r'medium grid - GPU')
g3 = top.errorbar(filter(local_gpu, 30)['hits'] / 30, filter(local_gpu, 30)['buildGridKernel'] / filter(local_gpu, 30)['hits'], yerr=filter(local_gpu, 30)['buildGridKernelVar']  / filter(local_gpu, 30)['hits'], fmt='go-', label=r'fine grid - GPU')

c1 = top.errorbar(filter(noLocal_cpu, 30)['hits'] / 30, filter(noLocal_cpu, 30)['buildGridKernel']  / filter(noLocal_cpu, 30)['hits'], yerr=filter(noLocal_cpu, 30)['buildGridKernelVar']  / filter(noLocal_cpu, 30)['hits'], fmt='bD:', label=r'coarse grid - CPU')
c2 = top.errorbar(filter(halfLocal_cpu, 30)['hits'] / 30, filter(halfLocal_cpu, 30)['buildGridKernel']  / filter(halfLocal_cpu, 30)['hits'], yerr=filter(halfLocal_cpu, 30)['buildGridKernelVar']  / filter(halfLocal_cpu, 30)['hits'], fmt='bx--', label=r'medium grid - CPU')
c3 = top.errorbar(filter(local_cpu, 30)['hits'] / 30, filter(local_cpu, 30)['buildGridKernel'] / filter(local_cpu, 30)['hits'], yerr=filter(local_cpu, 30)['buildGridKernelVar']  / filter(local_cpu, 30)['hits'], fmt='bo-', label=r'fine grid - CPU')

one = [1 for i in range(len(filter(halfLocal_gpu, 30)['hits']))]

bottom.plot(filter(halfLocal_gpu, 30)['hits'] / 30, one, 'ko-')

#bottom.plot(filter(local_gpu, 30)['hits'] / 30, filter(local_gpu, 30)['buildGridKernel'] / filter(local_gpu, 30)['buildGridKernel'], 'go-', label=r'coarse grid - GPU')
bottom.plot(filter(halfLocal_gpu, 30)['hits'] / 30, filter(local_gpu, 30)['buildGridKernel'] / filter(halfLocal_gpu, 30)['buildGridKernel'], 'gx--', label=r'medium grid - GPU')
bottom.plot(filter(noLocal_gpu, 30)['hits'] / 30, filter(local_gpu, 30)['buildGridKernel'] / filter(noLocal_gpu, 30)['buildGridKernel'], 'gD:', label=r'fine grid - GPU')

#bottom.plot(filter(local_cpu, 30)['hits'] / 30, filter(local_cpu, 30)['buildGridKernel'] / filter(local_cpu, 30)['buildGridKernel'], 'bo-', label=r'coarse grid - CPU')
bottom.plot(filter(halfLocal_cpu, 30)['hits'] / 30, filter(local_cpu, 30)['buildGridKernel'] / filter(halfLocal_cpu, 30)['buildGridKernel'], 'bx--', label=r'medium grid - CPU')
bottom.plot(filter(noLocal_cpu, 30)['hits'] / 30, filter(local_cpu, 30)['buildGridKernel'] / filter(noLocal_cpu, 30)['buildGridKernel'], 'bD:', label=r'fine grid - CPU')

top.set_title('Grid data structure construction')
bottom.set_xlabel('hits / event')
top.set_ylabel(r'time [ms]')
bottom.set_ylabel(r'ratio')
#plt.ylim(ymax=20)
bottom.set_xscale('log', basex=10)
top.set_xscale('log', basex=10)
top.set_yscale("log", basey=10)
bottom.set_ylim(0.2, 1.1)

l1 = [g1,g2,g3]
a = top.legend(l1, [x.get_label() for x in l1], loc=3)

l2 = [c1,c2,c3]
top.legend(l2, [x.get_label() for x in l2], loc=1)
top.add_artist(a)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig('../output/runtime/runtime_buildGrid.pdf')

#plt.show()
