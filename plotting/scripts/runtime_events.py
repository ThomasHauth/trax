import numpy as np

import latexHeader

import matplotlib.pyplot as plt

def filter(data, events):
    return data[data['events'] == events]

small_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.100.test.gpu.csv", delimiter = " ", names=True)
big_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.1000.test.gpu.csv", delimiter = " ", names=True)

small_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.100.test.cpu.csv", delimiter = " ", names=True)
big_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.events.1000.test.cpu.csv", delimiter = " ", names=True)

data_gpu = [(small_gpu, '100'), (big_gpu, '1000')]
data_cpu = [(small_cpu, '100'), (big_cpu, '1000')]

###############################
######## GPU

for item in data_gpu:

    data = item[0]
    
    fig = plt.figure(figsize=(11,6))
    ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))
    #ax2 = fig.add_subplot(111, position=(0.075, 0.10, 0.65, 0.22), sharex=ax)
    
    total = data['totalKernel'] + data['readTime'] + data['writeTime']
    
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
    
    ax.stackplot(data['events'], 
                  data['readTime'] / total, 
                  data['writeTime'] / total,
                  data['buildGridKernel'] / total, 
                  data['pairGenKernel'] / total, 
                  (data['tripletPredictCount'] + data['tripletPredictScan']) / total, 
                  data['tripletPredictStore'] / total,
                  data['tripletFilterKernel'] / total,
                  colors = colors)
    
    #ax2.plot(data['events'] / evt, total / evt, 'go-')
    
    #ax2.set_xlabel('events')
    ax.set_xlabel('events')
    ax.set_ylabel('runtime share [%]')
    ax.set_title('Composition of Event Processing Runtime - GPU')
    
    ax.set_xlim(np.min(data['events']), np.max(data['events']))
    ax.set_xscale('log', basex=2)
    
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")
    
    #ax2.set_xlim(np.min(data['events']) / evt, np.max(data['events']) / evt)
    #ax2.set_xscale('log', basex=2)
    #ax2.set_ylabel(r'time / event [ms]')
    
    ax.set_ylim(0,1)
    
    a = ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
    
    #a.set_title(r'\textbf{Tracks per event: %s}'%item[1])
    fig.text(0.725, 0.905, "Tracks: %s"%item[1], va='bottom', ha='right')
    
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.savefig('../output/runtime/contrib_events_%s_events.gpu.pdf'%item[1])
        
###############################
######## CPU

for item in data_cpu:

    data = item[0]
    
    fig = plt.figure(figsize=(11,6))
    ax = fig.add_subplot(111, position=(0.075, 0.1, 0.65, 0.8))
    #ax2 = fig.add_subplot(111, position=(0.075, 0.10, 0.65, 0.22), sharex=ax)
    
    total = data['totalKernel'] + data['readTime'] + data['writeTime']
    
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
    
    ax.stackplot(data['events'], 
                  data['readTime'] / total, 
                  data['writeTime'] / total,
                  data['buildGridKernel'] / total, 
                  data['pairGenKernel'] / total, 
                  (data['tripletPredictCount'] + data['tripletPredictScan']) / total, 
                  data['tripletPredictStore'] / total,
                  data['tripletFilterKernel'] / total,
                  colors = colors)
    
    #ax2.plot(data['events'] / evt, total / evt, 'go-')
    
    #ax2.set_xlabel('events')
    ax.set_xlabel('events')
    ax.set_ylabel('runtime share [%]')
    ax.set_title('Composition of Event Processing Runtime - CPU')
    
    ax.set_xlim(np.min(data['events']), np.max(data['events']))
    ax.set_xscale('log', basex=2)
    
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")
    
    #ax2.set_xlim(np.min(data['events']) / evt, np.max(data['events']) / evt)
    #ax2.set_xscale('log', basex=2)
    #ax2.set_ylabel(r'time / event [ms]')
    
    ax.set_ylim(0,1)
    
    a = ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
    
    #a.set_title(r'\textbf{Tracks per event: %s}'%item[1])
    fig.text(0.725, 0.905, "Tracks: %s"%item[1], va='bottom', ha='right')
    
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.savefig('../output/runtime/contrib_events_%s_events.cpu.pdf'%item[1])
        
###################################################
# kernel time for everything
fig = plt.figure()

gw1 = plt.errorbar(small_gpu['events'], small_gpu['totalWalltime'], yerr=small_gpu['totalWalltimeVar'], fmt='go--', label=r'GPU wall time')
gk1 = plt.errorbar(small_gpu['events'], small_gpu['totalKernel'], yerr=small_gpu['totalKernelVar'], fmt='go-', label=r'GPU kernel time')

cw1 = plt.errorbar(small_cpu['events'], small_cpu['totalWalltime'], yerr=small_cpu['totalWalltimeVar'], fmt='bo--', label=r'CPU wall time')
ck1 = plt.errorbar(small_cpu['events'], small_cpu['totalKernel'], yerr=small_cpu['totalKernelVar'], fmt='bo-', label=r'CPU kernel time')

plt.title('Processing Time for Concurrent Events')

plt.ylabel(r'time [ms]')
plt.yscale('log')

plt.xscale('log', basex=2)
plt.xlabel('events')

#l = [c1,g1,c2,g2,c3,g3]

#plt.legend(l, [x.get_label() for x in l], ncol=3, loc=2, mode='expand', columnspacing=0.8)
#plt.ylim(ymax = 20000)
a = plt.legend(ncol=2, loc=2)
#a.set_title('Tracks per event: 100')
#a.get_title().set_position((-145,0))

fig.text(0.9, 0.905, "Tracks: 100", va='bottom', ha='right')

#l1 = [g1,g2,g3]
#a = plt.legend(l1, [x.get_label() for x in l1], loc=2)
#
#l2 = [c1,c2,c3]
#plt.legend(l2, [x.get_label() for x in l2], loc=9)
#plt.add_artist(a)

plt.savefig('../output/runtime/runtime_events.100.pdf')

###################################################
# kernel time for everything
fig = plt.figure()


gw2 = plt.errorbar(big_gpu['events'], big_gpu['totalWalltime'], yerr=big_gpu['totalWalltimeVar'], fmt='go--', label=r'GPU wall time')
gk2 = plt.errorbar(big_gpu['events'], big_gpu['totalKernel'], yerr=big_gpu['totalKernelVar'], fmt='go-', label=r'GPU kernel time')

cw2 = plt.errorbar(big_cpu['events'], big_cpu['totalWalltime'], yerr=big_cpu['totalWalltimeVar'], fmt='bo--', label=r'CPU wall time')
ck2 = plt.errorbar(big_cpu['events'], big_cpu['totalKernel'], yerr=big_cpu['totalKernelVar'], fmt='bo-', label=r'CPU kernel time')

plt.title('Processing Time for Concurrent Events')

plt.ylabel(r'time [ms]')
plt.yscale('log')
#plt.ylim(10**0,10**6)

plt.xscale('log', basex=2)
plt.xlabel('events')


#l = [c1,g1,c2,g2,c3,g3]

#plt.legend(l, [x.get_label() for x in l], ncol=3, loc=2, mode='expand', columnspacing=0.8)
#plt.ylim(ymax = 20000)
a = plt.legend(ncol=2, loc=2)
#a.set_title('Tracks per event: 1000')
#a.get_title().set_position((-140,0))
fig.text(0.9, 0.905, "Tracks: 1000", va='bottom', ha='right')
#l1 = [g1,g2,g3]
#a = plt.legend(l1, [x.get_label() for x in l1], loc=2)
#
#l2 = [c1,c2,c3]
#plt.legend(l2, [x.get_label() for x in l2], loc=9)
#plt.add_artist(a)

plt.savefig('../output/runtime/runtime_events.1000.pdf')

###################################################
# kernel time per event for everything
fig = plt.figure()

gw1 = plt.errorbar(small_gpu['events'], small_gpu['totalWalltime'] / small_gpu['events'], yerr=small_gpu['totalWalltimeVar'] / small_gpu['events'], fmt='go--', label=r'GPU wall time')
gk1 = plt.errorbar(small_gpu['events'], small_gpu['totalKernel'] / small_gpu['events'], yerr=small_gpu['totalKernelVar'] / small_gpu['events'], fmt='go-', label=r'GPU kernel time')

cw1 = plt.errorbar(small_cpu['events'], small_cpu['totalWalltime'] / small_cpu['events'], yerr=small_cpu['totalWalltimeVar'] / small_cpu['events'], fmt='bo--', label=r'CPU wall time')
ck1 = plt.errorbar(small_cpu['events'], small_cpu['totalKernel'] / small_cpu['events'], yerr=small_cpu['totalKernelVar'] / small_cpu['events'], fmt='bo-', label=r'CPU kernel time')

plt.title('Processing Time for Concurrent Events')
plt.ylabel(r'time / event [ms]')
plt.yscale('log')

plt.xscale('log', basex=2)
plt.xlabel('events')

#l = [c1,g1,c2,g2,c3,g3]

#plt.legend(l, [x.get_label() for x in l], ncol=3, loc=2, mode='expand', columnspacing=0.8)
a = plt.legend(ncol=2, loc=1, columnspacing=1)
#a.set_title('Tracks per event: 100')
#a.get_title().set_position((-135,0))
fig.text(0.9, 0.905, "Tracks: 100", va='bottom', ha='right')
#l1 = [g1,g2,g3]
#a = plt.legend(l1, [x.get_label() for x in l1], loc=2)
#
#l2 = [c1,c2,c3]
#plt.legend(l2, [x.get_label() for x in l2], loc=9)
#plt.add_artist(a)

plt.savefig('../output/runtime/runtime_per_event.100.pdf')

###################################################
# kernel time per event for everything
fig = plt.figure()

gw1 = plt.errorbar(big_gpu['events'], big_gpu['totalWalltime'] / big_gpu['events'], yerr=big_gpu['totalWalltimeVar'] / big_gpu['events'], fmt='go--', label=r'GPU wall time')
gk1 = plt.errorbar(big_gpu['events'], big_gpu['totalKernel'] / big_gpu['events'], yerr=big_gpu['totalKernelVar'] / big_gpu['events'], fmt='go-', label=r'GPU kernel time')

cw1 = plt.errorbar(big_cpu['events'], big_cpu['totalWalltime'] / big_cpu['events'], yerr=big_cpu['totalWalltimeVar'] / big_cpu['events'], fmt='bo--', label=r'CPU wall time')
ck1 = plt.errorbar(big_cpu['events'], big_cpu['totalKernel'] / big_cpu['events'], yerr=big_cpu['totalKernelVar'] / big_cpu['events'], fmt='bo-', label=r'CPU kernel time')


plt.title('Processing Time for Concurrent Events')
plt.ylabel(r'time / event [ms]')
plt.yscale('log')
#plt.ylim(0.5, 10**3)

plt.xscale('log', basex=2)
plt.xlabel('events')

#l = [c1,g1,c2,g2,c3,g3]

#plt.legend(l, [x.get_label() for x in l], ncol=3, loc=2, mode='expand', columnspacing=0.8)
a = plt.legend(ncol=2, loc=3, columnspacing=1)
#a.set_title('Tracks per event: 1000')
#a.get_title().set_position((-130,0))
fig.text(0.9, 0.905, "Tracks: 1000", va='bottom', ha='right')
#l1 = [g1,g2,g3]
#a = plt.legend(l1, [x.get_label() for x in l1], loc=2)
#
#l2 = [c1,c2,c3]
#plt.legend(l2, [x.get_label() for x in l2], loc=9)
#plt.add_artist(a)

plt.savefig('../output/runtime/runtime_per_event.1000.pdf')