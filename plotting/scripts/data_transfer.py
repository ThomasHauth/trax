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

small_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.100.test.gpu.csv", delimiter = " ", names=True)
big_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.1000.test.gpu.csv", delimiter = " ", names=True)

small_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.100.test.cpu.csv", delimiter = " ", names=True)
big_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.1000.test.cpu.csv", delimiter = " ", names=True)

evt = 30

###################################################
# data transferred (tracks)
fig = plt.figure()

bw = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
vol = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=bw)

bw.set_title(r'Data Transfer per Track')

g1 = bw.plot(filter(local_gpu,evt)['tracks'] / evt, (filter(local_gpu,evt)['writeBytes'] / (10**6 * filter(local_gpu,evt)['writeTime'])), 'go-', label='write - GPU')
g2 = bw.plot(filter(local_gpu,evt)['tracks'] / evt, (filter(local_gpu,evt)['readBytes'] / (10**6 * filter(local_gpu,evt)['readTime'])), 'gx--', label='read - GPU')

c1 = bw.plot(filter(local_cpu,evt)['tracks'] / evt, (filter(local_cpu,evt)['writeBytes'] / (10**6 * filter(local_cpu,evt)['writeTime'])), 'bo-', label='write - CPU')
c2 = bw.plot(filter(local_cpu,evt)['tracks'] / evt, (filter(local_cpu,evt)['readBytes'] / (10**6 * filter(local_cpu,evt)['readTime'])), 'bx--', label='read - CPU')


bw1 = vol.plot(filter(local_gpu,evt)['tracks'] / evt, (filter(local_gpu,evt)['writeBytes'] / 10**6 )  / evt, 'ko-', label='written - coarse grid')
bw2 = vol.plot(filter(noLocal_gpu,evt)['tracks'] / evt, (filter(noLocal_gpu,evt)['writeBytes'] / 10**6 )  / evt, 'yo-', label='written - fine grid')

br1 = vol.plot(filter(local_gpu,evt)['tracks'] / evt, (filter(local_gpu,evt)['readBytes'] / 10**6 ) / evt, 'kx--', label='read - coarse grid')
br2 = vol.plot(filter(noLocal_gpu,evt)['tracks'] / evt, (filter(noLocal_gpu,evt)['readBytes'] / 10**6 ) / evt, 'yx--', label='read - fine grid')

bw.set_ylabel('bandwith [GB/s]')
bw.set_xscale('log', basex=2)

vol.set_xlabel('tracks')
vol.set_xscale('log', basex=2)
vol.set_ylabel(r'[MB]')
vol.set_yscale('log')

l1 = [g1[0], g2[0],c1[0], c2[0]]
l2 = [bw1[0], br1[0], bw2[0], br2[0]]

#bw.set_ylim(3,10)
vol.set_ylim(ymax=200)
bw.legend(l1, [x.get_label() for x in l1], ncol=2, loc=2, columnspacing=0.8)
vol.legend(l2, [x.get_label() for x in l2], ncol=2, loc=2, columnspacing=0.8)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig('../output/runtime/dataTransfer_tracks.pdf')

####################################################
## data transferred (events)
fig = plt.figure()

bw = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
vol = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=bw)

bw.set_title(r'Data Transfer for Concurrent Events')

g1 = bw.plot(small_gpu['events'], (small_gpu['writeBytes'] / (10**6 * small_gpu['writeTime'])), 'go-', label='write - GPU')
g2 = bw.plot(small_gpu['events'], (small_gpu['readBytes'] / (10**6 * small_gpu['readTime'])), 'gx--', label='read - GPU')

c1 = bw.plot(small_cpu['events'], (small_cpu['writeBytes'] / (10**6 * small_cpu['writeTime'])), 'bo-', label='write - CPU')
c2 = bw.plot(small_cpu['events'], (small_cpu['readBytes'] / (10**6 * small_cpu['readTime'])), 'bx--', label='read - CPU')


bw1 = vol.plot(small_gpu['events'], (small_gpu['writeBytes'] / 10**6 )  / evt, 'ko-', label='written - 100 tracks')
bw2 = vol.plot(big_gpu['events'], (big_gpu['writeBytes'] / 10**6 )  / evt, 'yo-', label='written - 1000 tracks')

br1 = vol.plot(small_gpu['events'], (small_gpu['readBytes'] / 10**6 ) / evt, 'kx--', label='read - 100 tracks')
br2 = vol.plot(big_gpu['events'], (big_gpu['readBytes'] / 10**6 ) / evt, 'yx--', label='read - 1000 tracks')

bw.set_ylabel('bandwith [GB/s]')
bw.set_xscale('log', basex=2)

vol.set_xlabel('events')
vol.set_xscale('log', basex=2)
vol.set_ylabel(r'[MB]')
vol.set_yscale('log')

vol.set_ylim(10**-5,30)

l1 = [g1[0], g2[0],c1[0], c2[0]]
l2 = [bw1[0], br1[0], bw2[0], br2[0]]

bw.legend(l1, [x.get_label() for x in l1], ncol=2, loc=4, columnspacing=0.8)
vol.legend(l2, [x.get_label() for x in l2], ncol=2, loc=4, columnspacing=0.8, borderpad=0.2)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig('../output/runtime/dataTransfer_events.pdf')

