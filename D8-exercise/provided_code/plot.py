import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# nfiles = ["dsyevd.dat", "dsyevd_gpu.dat", "dsyevd_m.dat"] # data files
nfiles = ["scalapack_timing.dat", "plasma_timing.dat", "magma_timing.dat"] # data files
rep = 6     # number of repetition for each matrix size

fig, ax = plt.subplots() # create a new figure with a default 111 subplot

for namef in nfiles:
    data1 = np.loadtxt(namef)
    size_m = data1[:,0]
    sec1   = data1[:,1]

    if sec1[-1] < 1:
        size_m = size_m[:-rep]
        sec1 = sec1[:-rep]

    lvec = len(sec1) / rep

    s = np.zeros(lvec)
    t1 = np.zeros(lvec)
    err1 = np.zeros(lvec)

    i = 0
    j = 0
    count  = 0
    t_tmp1 = 0.

    while i < len(sec1):
        while j < rep:
            # performing mean value calculation
            t_tmp1 += sec1[i + j]
            j += 1


        s[count] = size_m[i]
        t1[count] = t_tmp1 / rep

        t_tmp1 = 0.
        j -= rep # need to perform error calculation
        
        while j < rep:
            # performing error calculation
            t_tmp1 += (sec1[i + j] - t1[count])**2
            j += 1

        err1[count] = (t_tmp1 / (rep - 1.))**0.5

        t_tmp1 = 0.
        count += 1
        i += rep
        j = 0
        
    ax.errorbar(s, t1, yerr=err1, label=namef[:-11].upper()) 

ax.set_xlabel('Matrix Size')
ax.set_ylabel('Time (s)')
# ax.set_title('MAGMA Execution Time (ULISSE)')
ax.set_title('DSYEV with ScaLAPACK - PLASMA - MAGMA (Execution Time on ULISSE)\n')
ax.legend(bbox_to_anchor = (.34, 1.))
# plt.show()
plt.savefig('final_comparison.png')
