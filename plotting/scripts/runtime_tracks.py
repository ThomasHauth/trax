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

data_gpu = [(local_gpu, 'coarse grid'), (halfLocal_gpu, 'medium grid'), (noLocal_gpu, 'fine grid')]
data_cpu = [(local_cpu, 'coarse grid'), (halfLocal_cpu, 'medium grid'), (noLocal_cpu, 'fine grid')]
evts = [1,30]

###############################
######## GPU

for evt in evts:
    for item in data_gpu:

        data = filter(item[0], evt)
        
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
        
        ax.stackplot(data['tracks'] / evt, 
                      data['readTime'] / total, 
                      data['writeTime'] / total,
                      data['buildGridKernel'] / total, 
                      data['pairGenKernel'] / total, 
                      (data['tripletPredictCount'] + data['tripletPredictScan']) / total, 
                      data['tripletPredictStore'] / total,
                      data['tripletFilterKernel'] / total,
                      colors = colors)
        
        #ax2.plot(data['tracks'] / evt, total / evt, 'go-')
        
        #ax2.set_xlabel('tracks')
        ax.set_xlabel('tracks')
        ax.set_ylabel('runtime share [%]')
        ax.set_title('Composition of Event Processing Runtime - GPU')
        
        ax.set_xlim(np.min(data['tracks']) / evt, np.max(data['tracks']) / evt)
        ax.set_xscale('log', basex=2)
        
        #ax2.yaxis.tick_right()
        #ax2.yaxis.set_label_position("right")
        
        #ax2.set_xlim(np.min(data['tracks']) / evt, np.max(data['tracks']) / evt)
        #ax2.set_xscale('log', basex=2)
        #ax2.set_ylabel(r'time / event [ms]')
        
        ax.set_ylim(0,1)
        
        a = ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
        
        #a.set_title(r'\textbf{Events: %s $\qquad$ %s}'%(str(evt), item[1]))
        fig.text(0.075, 0.905, "%s"%item[1], va='bottom', ha='left')
        fig.text(0.725, 0.905, "Events: %s"%str(evt), va='bottom', ha='right')
        
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.savefig('../output/runtime/contrib_tracks_%s_%s.gpu.pdf'%(str(evt), item[1].replace (" ", "_")))
        
###############################
######## CPU

for evt in evts:
    for item in data_cpu:

        data = filter(item[0], evt)
        
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
        
        ax.stackplot(data['tracks'] / evt, 
                      data['readTime'] / total, 
                      data['writeTime'] / total,
                      data['buildGridKernel'] / total, 
                      data['pairGenKernel'] / total, 
                      (data['tripletPredictCount'] + data['tripletPredictScan']) / total, 
                      data['tripletPredictStore'] / total,
                      data['tripletFilterKernel'] / total,
                      colors = colors)
        
        #ax2.plot(data['tracks'] / evt, total / evt, 'go-')
        
        #ax2.set_xlabel('tracks')
        ax.set_xlabel('tracks')
        ax.set_ylabel('runtime share [%]')
        ax.set_title('Composition of Event Processing Runtime - CPU')
        
        ax.set_xlim(np.min(data['tracks']) / evt, np.max(data['tracks']) / evt)
        ax.set_xscale('log', basex=2)
        
        #ax2.yaxis.tick_right()
        #ax2.yaxis.set_label_position("right")
        
        #ax2.set_xlim(np.min(data['tracks']) / evt, np.max(data['tracks']) / evt)
        #ax2.set_xscale('log', basex=2)
        #ax2.set_ylabel(r'time / event [ms]')
        
        ax.set_ylim(0,1)
        
        a = ax.legend(legendHandles[::-1], labels[::-1], bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0, 0, 1,1), loc=5)
        
        #a.set_title(r'\textbf{Events: %s $\qquad$ %s}'%(str(evt), item[1]))
        fig.text(0.075, 0.905, "%s"%item[1], va='bottom', ha='left')
        fig.text(0.725, 0.905, "Events: %s"%str(evt), va='bottom', ha='right')
        
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.savefig('../output/runtime/contrib_tracks_%s_%s.cpu.pdf'%(str(evt), item[1].replace (" ", "_")))
        
###################################################
# kernel time for everything
fig = plt.figure()
top = fig.add_subplot(111, position=[0.11, 0.33, 0.86, 0.60])
bottom = fig.add_subplot(111, position=[0.11, 0.10, 0.86, 0.22], sharex=top)

g1 = top.errorbar(filter(local_gpu, 30)['tracks'] / 30, filter(local_gpu, 30)['totalKernel'] / filter(local_gpu, 30)['tracks'], yerr=filter(local_gpu, 30)['totalKernelVar']  / filter(local_gpu, 30)['hits'], fmt='go-', label=r'coarse grid - GPU')
g2 = top.errorbar(filter(halfLocal_gpu, 30)['tracks'] / 30, filter(halfLocal_gpu, 30)['totalKernel'] / filter(halfLocal_gpu, 30)['tracks'], yerr=filter(local_gpu, 30)['totalKernelVar']  / filter(local_gpu, 30)['hits'], fmt='gx--', label=r'medium grid - GPU')
g3 = top.errorbar(filter(noLocal_gpu, 30)['tracks'] / 30, filter(noLocal_gpu, 30)['totalKernel']  / filter(noLocal_gpu, 30)['tracks'], yerr=filter(noLocal_gpu, 30)['totalKernelVar']  / filter(noLocal_gpu, 30)['hits'], fmt='gD:', label=r'fine grid - GPU')

c1 = top.errorbar(filter(local_cpu, 30)['tracks'] / 30, filter(local_cpu, 30)['totalKernel'] / filter(local_cpu, 30)['tracks'], yerr=filter(local_cpu, 30)['totalKernelVar']  / filter(local_cpu, 30)['hits'], fmt='bo-', label=r'coarse grid - CPU')
c2 = top.errorbar(filter(halfLocal_cpu, 30)['tracks'] / 30, filter(halfLocal_cpu, 30)['totalKernel'] / filter(halfLocal_cpu, 30)['tracks'], yerr=filter(local_cpu, 30)['totalKernelVar']  / filter(local_cpu, 30)['hits'], fmt='bx--', label=r'medium grid - CPU')
c3 = top.errorbar(filter(noLocal_cpu, 30)['tracks'] / 30, filter(noLocal_cpu, 30)['totalKernel']  / filter(noLocal_cpu, 30)['tracks'], yerr=filter(noLocal_cpu, 30)['totalKernelVar']  / filter(noLocal_cpu, 30)['hits'], fmt='bD:', label=r'fine grid - CPU')

one = [1 for i in range(len(filter(halfLocal_gpu, 30)['tracks']))]
bottom.plot(filter(halfLocal_gpu, 30)['tracks'] / 30, one, 'ko-')

#bottom.plot(filter(local_gpu, 30)['tracks'] / 30, filter(local_gpu, 30)['totalKernel'] / filter(local_gpu, 30)['totalKernel'], 'go-', label=r'coarse grid - GPU')
bottom.plot(filter(halfLocal_gpu, 30)['tracks'] / 30, filter(local_gpu, 30)['totalKernel'] / filter(halfLocal_gpu, 30)['totalKernel'], 'gx--', label=r'medium grid - GPU')
bottom.plot(filter(noLocal_gpu, 30)['tracks'] / 30, filter(local_gpu, 30)['totalKernel'] / filter(noLocal_gpu, 30)['totalKernel'], 'gD:', label=r'fine grid - GPU')

#bottom.plot(filter(local_cpu, 30)['tracks'] / 30, filter(local_cpu, 30)['totalKernel'] / filter(local_cpu, 30)['totalKernel'], 'bo-', label=r'coarse grid - CPU')
bottom.plot(filter(halfLocal_cpu, 30)['tracks'] / 30, filter(local_cpu, 30)['totalKernel'] / filter(halfLocal_cpu, 30)['totalKernel'], 'bx--', label=r'medium grid - CPU')
bottom.plot(filter(noLocal_cpu, 30)['tracks'] / 30, filter(local_cpu, 30)['totalKernel'] / filter(noLocal_cpu, 30)['totalKernel'], 'bD:', label=r'fine grid - CPU')


top.set_title('Processing Time over Tracks')
bottom.set_xlabel('tracks / event')
top.set_ylabel(r'time [ms]')
top.set_ylim(10**-4, 10**2)
bottom.set_ylabel(r'ratio')
#plt.ylim(ymax=20)
bottom.set_xscale('log', basex=10)
top.set_xscale('log', basex=10)
top.set_yscale("log", basey=10)

l = [c1,g1,c2,g2,c3,g3]

top.legend(l, [x.get_label() for x in l], ncol=3, loc=2, mode='expand', columnspacing=0.8)

#l1 = [g1,g2,g3]
#a = top.legend(l1, [x.get_label() for x in l1], loc=2)
#
#l2 = [c1,c2,c3]
#top.legend(l2, [x.get_label() for x in l2], loc=9)
#top.add_artist(a)

plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig('../output/runtime/runtime_tracks.pdf')