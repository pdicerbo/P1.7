import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

nfiles = ["1x20timing.dat", "5x4timing.dat", "4x5timing.dat", "2x10timing.dat", "10x2timing.dat"]
rep = 10     # number of repetition for each matrix size

# plt.figure()
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
axins = zoomed_inset_axes(ax, 2.8, loc=2) # zoom-factor: 2.5, location: upper-left

for namef in nfiles:
    data1 = np.loadtxt(namef)
    size_m = data1[:,0]
    sec1   = data1[:,1]

    lvec = len(data1) / rep

    s = np.zeros(lvec)
    t1 = np.zeros(lvec)
    err1 = np.zeros(lvec)

    i = 0
    j = 0
    count  = 0
    t_tmp1 = 0.

    while i < len(sec1):
        while j < rep:
            t_tmp1 += sec1[i + j]
            j += 1


        s[count] = size_m[i]
        t1[count] = t_tmp1 / rep

        t_tmp1 = 0.
        j -= rep
        
        while j < rep:
            t_tmp1 += (sec1[i + j] - t1[count])**2
            j += 1

        err1[count] = (t_tmp1 / (rep - 1.))**0.5

        t_tmp1 = 0.
        count += 1
        i += rep
        j = 0

    ax.errorbar(s, t1, yerr=err1, label=namef[:-10])
    axins.errorbar(s, t1, yerr=err1, label=namef[:-10])
    
axins.set_xlim(4000, 8000)
axins.set_ylim(2, 28)
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.axis([4000, 18000, 0, 300])
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Time (s)')
ax.set_title('PDSYEV Execution Time (ULISSE)')
ax.legend(title = 'Grid structure:', bbox_to_anchor = (1., .42))
# plt.show()
plt.savefig('pdsyev_timing.png')
