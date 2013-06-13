import numpy as np

import latexHeader

import matplotlib.pyplot as plt

def filter(data, events, tracks):
    return data[data['tracks'] / events == tracks]

wgSize_gpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.workGroupSize.test.gpu.csv", delimiter = " ", names=True)
wgSize_cpu = np.genfromtxt("../data/runtime/runtime.bigOne.local.conf.workGroupSize.test.cpu.csv", delimiter = " ", names=True)

###############################
######## GPU

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_gpu, 1, 100)['threads'], filter(wgSize_gpu, 1, 100)['totalKernel'], yerr=filter(wgSize_gpu, 1, 100)['totalKernelVar'], fmt='go-', label='total GPU - 1:100')
plt.errorbar(filter(wgSize_gpu, 1, 1000)['threads'], filter(wgSize_gpu, 1, 1000)['totalKernel'], yerr=filter(wgSize_gpu, 1, 1000)['totalKernelVar'], fmt='go--', label='total GPU - 1:1000')

plt.errorbar(filter(wgSize_gpu, 50, 100)['threads'], filter(wgSize_gpu, 50, 100)['totalKernel'] / 50, yerr=filter(wgSize_gpu, 50, 100)['totalKernelVar'] / 50, fmt='gx-', label='total GPU - 50:100')
plt.errorbar(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_gpu, 50, 1000)['totalKernel'] / 50, yerr=filter(wgSize_gpu, 50, 1000)['totalKernelVar'] / 50, fmt='gx--', label='total GPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
plt.ylim(ymax=20)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.total.gpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_gpu, 1, 100)['threads'], filter(wgSize_gpu, 1, 100)['buildGridKernel'], yerr=filter(wgSize_gpu, 1, 100)['buildGridKernelVar'], fmt='go-', label='build grid GPU - 1:100')
plt.errorbar(filter(wgSize_gpu, 1, 1000)['threads'], filter(wgSize_gpu, 1, 1000)['buildGridKernel'], yerr=filter(wgSize_gpu, 1, 1000)['buildGridKernelVar'], fmt='go--', label='build grid GPU - 1:1000')

plt.errorbar(filter(wgSize_gpu, 50, 100)['threads'], filter(wgSize_gpu, 50, 100)['buildGridKernel'] / 50, yerr=filter(wgSize_gpu, 50, 100)['buildGridKernelVar'] / 50, fmt='gx-', label='build grid GPU - 50:100')
plt.errorbar(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_gpu, 50, 1000)['buildGridKernel'] / 50, yerr=filter(wgSize_gpu, 50, 1000)['buildGridKernelVar'] / 50, fmt='gx--', label='build grid GPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
plt.ylim(ymax=1.5)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.buildGrid.gpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_gpu, 1, 100)['threads'], filter(wgSize_gpu, 1, 100)['pairGenKernel'], yerr=filter(wgSize_gpu, 1, 100)['pairGenKernelVar'], fmt='go-', label='pair gen GPU - 1:100')
plt.errorbar(filter(wgSize_gpu, 1, 1000)['threads'], filter(wgSize_gpu, 1, 1000)['pairGenKernel'], yerr=filter(wgSize_gpu, 1, 1000)['pairGenKernelVar'], fmt='go--', label='pair gen GPU - 1:1000')

plt.errorbar(filter(wgSize_gpu, 50, 100)['threads'], filter(wgSize_gpu, 50, 100)['pairGenKernel'] / 50, yerr=filter(wgSize_gpu, 50, 100)['pairGenKernelVar'] / 50, fmt='gx-', label='pair gen GPU - 50:100')
plt.errorbar(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_gpu, 50, 1000)['pairGenKernel'] / 50, yerr=filter(wgSize_gpu, 50, 1000)['pairGenKernelVar'] / 50, fmt='gx--', label='pair gen GPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
plt.ylim(ymax=8)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.pairGen.gpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_gpu, 1, 100)['threads'], filter(wgSize_gpu, 1, 100)['tripletPredictKernel'], yerr=filter(wgSize_gpu, 1, 100)['tripletPredictKernelVar'], fmt='go-', label='triplet predict GPU - 1:100')
plt.errorbar(filter(wgSize_gpu, 1, 1000)['threads'], filter(wgSize_gpu, 1, 1000)['tripletPredictKernel'], yerr=filter(wgSize_gpu, 1, 1000)['tripletPredictKernelVar'], fmt='go--', label='triplet predict GPU - 1:1000')

plt.errorbar(filter(wgSize_gpu, 50, 100)['threads'], filter(wgSize_gpu, 50, 100)['tripletPredictKernel'] / 50, yerr=filter(wgSize_gpu, 50, 100)['tripletPredictKernelVar'] / 50, fmt='gx-', label='triplet predict GPU - 50:100')
plt.errorbar(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_gpu, 50, 1000)['tripletPredictKernel'] / 50, yerr=filter(wgSize_gpu, 50, 1000)['tripletPredictKernelVar'] / 50, fmt='gx--', label='triplet predict GPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
plt.ylim(ymax=5)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.tripletPredict.gpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_gpu, 1, 100)['threads'], filter(wgSize_gpu, 1, 100)['tripletFilterKernel'], yerr=filter(wgSize_gpu, 1, 100)['tripletFilterKernelVar'], fmt='go-', label='triplet filter GPU - 1:100')
plt.errorbar(filter(wgSize_gpu, 1, 1000)['threads'], filter(wgSize_gpu, 1, 1000)['tripletFilterKernel'], yerr=filter(wgSize_gpu, 1, 1000)['tripletFilterKernelVar'], fmt='go--', label='triplet filter GPU - 1:1000')

plt.errorbar(filter(wgSize_gpu, 50, 100)['threads'], filter(wgSize_gpu, 50, 100)['tripletFilterKernel'] / 50, yerr=filter(wgSize_gpu, 50, 100)['tripletFilterKernelVar'] / 50, fmt='gx-', label='triplet filter GPU - 50:100')
plt.errorbar(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_gpu, 50, 1000)['tripletFilterKernel'] / 50, yerr=filter(wgSize_gpu, 50, 1000)['tripletFilterKernelVar'] / 50, fmt='gx--', label='triplet filter GPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
plt.ylim(ymax=2)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.tripletFilter.gpu.pdf')

###########################
#### CPU

###################################################
# time for varying work-group sizes
f,(top,bottom) = plt.subplots(2,1,sharex=True)

top.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['totalKernel'], yerr=filter(wgSize_cpu, 1, 100)['totalKernelVar'], fmt='bo-', label='total CPU - 1:100')
bottom.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['totalKernel'], yerr=filter(wgSize_cpu, 1, 100)['totalKernelVar'], fmt='bo-', label='total CPU - 1:100')

top.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['totalKernel'], yerr=filter(wgSize_cpu, 1, 1000)['totalKernelVar'], fmt='bo--', label='total CPU - 1:1000')
bottom.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['totalKernel'], yerr=filter(wgSize_cpu, 1, 1000)['totalKernelVar'], fmt='bo--', label='total CPU - 1:1000')

top.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['totalKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['totalKernelVar'] / 50, fmt='bx-', label='total CPU - 50:100')
bottom.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['totalKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['totalKernelVar'] / 50, fmt='bx-', label='total CPU - 50:100')

top.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['totalKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['totalKernelVar'] / 50, fmt='bx--', label='total CPU - 50:1000')
bottom.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['totalKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['totalKernelVar'] / 50, fmt='bx--', label='total CPU - 50:1000')

top.set_ylim(250, 400)
bottom.set_ylim(0, 5.5)

top.spines['bottom'].set_visible(False)
bottom.spines['top'].set_visible(False)
top.xaxis.tick_top()
top.tick_params(labeltop='off') # don't put tick labels at the top
bottom.xaxis.tick_bottom()

top.set_title('Processing Time with Work-Group Size')
bottom.set_xlabel('work-group size')
#top.set_ylabel(r'time [ms]')

plt.text(0.05, 0.5, r'time [ms]', transform=plt.gcf().transFigure, rotation=90, va='center')

#plt.ylim(ymtop=20)
plt.xscale('log', basex=2)
#plt.yscale("log")

d = .015 # how big to make the diagonal lines in topes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=top.transAxes, color='k', clip_on=False)
top.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
top.plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

kwargs.update(transform=bottom.transAxes)  # switch to the bottom axes
bottom.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
bottom.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal

top.legend(loc=1)

plt.savefig('../output/runtime/runtime_wg_size.total.cpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['buildGridKernel'], yerr=filter(wgSize_cpu, 1, 100)['buildGridKernelVar'], fmt='bo-', label='build grid CPU - 1:100')
plt.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['buildGridKernel'], yerr=filter(wgSize_cpu, 1, 1000)['buildGridKernelVar'], fmt='bo--', label='build grid CPU - 1:1000')

plt.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['buildGridKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['buildGridKernelVar'] / 50, fmt='bx-', label='build grid CPU - 50:100')
plt.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['buildGridKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['buildGridKernelVar'] / 50, fmt='bx--', label='build grid CPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
#plt.ylim(ymax=1.5)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.buildGrid.cpu.pdf')

###################################################
# time for varying work-group sizes
plt.figure()
plt.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['pairGenKernel'], yerr=filter(wgSize_cpu, 1, 100)['pairGenKernelVar'], fmt='bo-', label='pair gen CPU - 1:100')
plt.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['pairGenKernel'], yerr=filter(wgSize_cpu, 1, 1000)['pairGenKernelVar'], fmt='bo--', label='pair gen CPU - 1:1000')

plt.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['pairGenKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['pairGenKernelVar'] / 50, fmt='bx-', label='pair gen CPU - 50:100')
plt.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['pairGenKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['pairGenKernelVar'] / 50, fmt='bx--', label='pair gen CPU - 50:1000')
plt.title('Processing Time with Work-Group Size')
plt.xlabel('work-group size')
plt.ylabel(r'time [ms]')
#plt.ylim(ymax=5)
plt.xscale('log', basex=2)
#plt.yscale("log")

plt.legend(loc=2)

plt.savefig('../output/runtime/runtime_wg_size.pairGen.cpu.pdf')

###################################################
# time for varying work-group sizes
f,(top,bottom) = plt.subplots(2,1,sharex=True)

top.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['tripletPredictKernel'], yerr=filter(wgSize_cpu, 1, 100)['tripletPredictKernelVar'], fmt='bo-', label='triplet predict CPU - 1:100')
top.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['tripletPredictKernel'], yerr=filter(wgSize_cpu, 1, 1000)['tripletPredictKernelVar'], fmt='bo--', label='triplet predict CPU - 1:1000')

top.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['tripletPredictKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['tripletPredictKernelVar'] / 50, fmt='bx-', label='triplet predict CPU - 50:100')
top.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['tripletPredictKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['tripletPredictKernelVar'] / 50, fmt='bx--', label='triplet predict CPU - 50:1000')

bottom.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['tripletPredictKernel'], yerr=filter(wgSize_cpu, 1, 100)['tripletPredictKernelVar'], fmt='bo-', label='triplet predict CPU - 1:100')
bottom.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['tripletPredictKernel'], yerr=filter(wgSize_cpu, 1, 1000)['tripletPredictKernelVar'], fmt='bo--', label='triplet predict CPU - 1:1000')

bottom.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['tripletPredictKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['tripletPredictKernelVar'] / 50, fmt='bx-', label='triplet predict CPU - 50:100')
bottom.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['tripletPredictKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['tripletPredictKernelVar'] / 50, fmt='bx--', label='triplet predict CPU - 50:1000')

top.set_ylim(190, 300)
bottom.set_ylim(0, 5.5)

top.spines['bottom'].set_visible(False)
bottom.spines['top'].set_visible(False)
top.xaxis.tick_top()
top.tick_params(labeltop='off') # don't put tick labels at the top
bottom.xaxis.tick_bottom()

top.set_title('Processing Time with Work-Group Size')
bottom.set_xlabel('work-group size')
#top.set_ylabel(r'time [ms]')

plt.text(0.05, 0.5, r'time [ms]', transform=plt.gcf().transFigure, rotation=90, va='center')

#plt.ylim(ymtop=20)
plt.xscale('log', basex=2)
#plt.yscale("log")

d = .015 # how big to make the diagonal lines in topes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=top.transAxes, color='k', clip_on=False)
top.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
top.plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

kwargs.update(transform=bottom.transAxes)  # switch to the bottom axes
bottom.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
bottom.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal

bottom.legend(loc=1)

plt.savefig('../output/runtime/runtime_wg_size.tripletPredict.cpu.pdf')

###################################################
# time for varying work-group sizes
f,(top,bottom) = plt.subplots(2,1,sharex=True)

top.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['tripletFilterKernel'], yerr=filter(wgSize_cpu, 1, 100)['tripletFilterKernelVar'], fmt='bo-', label='triplet filter CPU - 1:100')
top.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['tripletFilterKernel'], yerr=filter(wgSize_cpu, 1, 1000)['tripletFilterKernelVar'], fmt='bo--', label='triplet filter CPU - 1:1000')

top.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['tripletFilterKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['tripletFilterKernelVar'] / 50, fmt='bx-', label='triplet filter CPU - 50:100')
top.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['tripletFilterKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['tripletFilterKernelVar'] / 50, fmt='bx--', label='triplet filter CPU - 50:1000')

bottom.errorbar(filter(wgSize_cpu, 1, 100)['threads'], filter(wgSize_cpu, 1, 100)['tripletFilterKernel'], yerr=filter(wgSize_cpu, 1, 100)['tripletFilterKernelVar'], fmt='bo-', label='triplet filter CPU - 1:100')
bottom.errorbar(filter(wgSize_cpu, 1, 1000)['threads'], filter(wgSize_cpu, 1, 1000)['tripletFilterKernel'], yerr=filter(wgSize_cpu, 1, 1000)['tripletFilterKernelVar'], fmt='bo--', label='triplet filter CPU - 1:1000')

bottom.errorbar(filter(wgSize_cpu, 50, 100)['threads'], filter(wgSize_cpu, 50, 100)['tripletFilterKernel'] / 50, yerr=filter(wgSize_cpu, 50, 100)['tripletFilterKernelVar'] / 50, fmt='bx-', label='triplet filter CPU - 50:100')
bottom.errorbar(filter(wgSize_cpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['tripletFilterKernel'] / 50, yerr=filter(wgSize_cpu, 50, 1000)['tripletFilterKernelVar'] / 50, fmt='bx--', label='triplet filter CPU - 50:1000')


top.set_ylim(45, 100)
bottom.set_ylim(0, 1.5)

top.spines['bottom'].set_visible(False)
bottom.spines['top'].set_visible(False)
top.xaxis.tick_top()
top.tick_params(labeltop='off') # don't put tick labels at the top
bottom.xaxis.tick_bottom()

top.set_title('Processing Time with Work-Group Size')
bottom.set_xlabel('work-group size')
#top.set_ylabel(r'time [ms]')

plt.text(0.05, 0.5, r'time [ms]', transform=plt.gcf().transFigure, rotation=90, va='center')

#plt.ylim(ymtop=20)
plt.xscale('log', basex=2)
#plt.yscale("log")

d = .015 # how big to make the diagonal lines in topes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=top.transAxes, color='k', clip_on=False)
top.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
top.plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

kwargs.update(transform=bottom.transAxes)  # switch to the bottom axes
bottom.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
bottom.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal

bottom.legend(loc=1)

plt.savefig('../output/runtime/runtime_wg_size.tripletFilter.cpu.pdf')

###################################################
# speedup for varying work-group sizes
plt.figure()

wg1GPU = wgSize_gpu[(wgSize_gpu['threads'] == 1) & (wgSize_gpu['tracks'] == 50000)]['totalKernel']
wg1CPU = wgSize_cpu[(wgSize_cpu['threads'] == 1) & (wgSize_cpu['tracks'] == 50000)]['totalKernel']

plt.plot(filter(wgSize_gpu, 50, 1000)['threads'], wg1GPU/ filter(wgSize_gpu, 50, 1000)['totalKernel'], 'go-', label='GPU - relative')
plt.plot(filter(wgSize_gpu, 50, 1000)['threads'], wg1CPU/ filter(wgSize_gpu, 50, 1000)['totalKernel'], 'gx--', label='GPU - absolute')
plt.plot(filter(wgSize_gpu, 50, 1000)['threads'], filter(wgSize_cpu, 50, 1000)['totalKernel']/ filter(wgSize_gpu, 50, 1000)['totalKernel'], 'kd:', label='GPU over CPU')
plt.plot(filter(wgSize_cpu, 50, 1000)['threads'], wg1CPU/ filter(wgSize_cpu, 50, 1000)['totalKernel'], 'bo-', label='CPU - relative')

plt.title('Speedup for Varying Work-Group Sizes')
plt.xlabel('work-group size')
plt.ylabel(r'speedup')
#plt.xlim(np.min(threads_gpu['threads']), np.max(threads_gpu['threads']))
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)

plt.legend(loc=2)
plt.savefig('../output/runtime/speedup_wg_size.total.pdf')


###################################################
# efficiency for varying work-group sizes
plt.figure()

wg1GPU = wgSize_gpu[(wgSize_gpu['threads'] == 1) & (wgSize_gpu['tracks'] == 50000)]['totalKernel']
wg1CPU = wgSize_cpu[(wgSize_cpu['threads'] == 1) & (wgSize_cpu['tracks'] == 50000)]['totalKernel']

plt.plot(filter(wgSize_gpu, 50, 1000)['threads'], (wg1GPU/ filter(wgSize_gpu, 50, 1000)['totalKernel']) / filter(wgSize_gpu, 50, 1000)['threads'], 'go-', label='GPU - relative')
plt.plot(filter(wgSize_cpu, 50, 1000)['threads'], (wg1CPU/ filter(wgSize_cpu, 50, 1000)['totalKernel']) / filter(wgSize_cpu, 50, 1000)['threads'], 'bo-', label='CPU - relative')

plt.title('Efficiency for Varying Work-Group Sizes')
plt.xlabel('work-group size')
plt.ylabel(r'efficiency')
plt.xscale('log', basex=2)

plt.legend()

plt.savefig('../output/runtime/efficiency_wg_size.total.pdf')