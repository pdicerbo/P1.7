import numpy as np
import matplotlib.pyplot as plt

nfiles = ["12x2timing.dat", "1x24timing.dat", "2x12timing.dat", "3x8timing.dat", "8x3timing.dat", "6x4timing.dat"]
rep = 10     # number of repetition for each matrix size

plt.figure()

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

    plt.errorbar(s, t1, yerr=err1, label=namef[:-10])

plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title('Execution Time (COSINT)')
plt.legend(bbox_to_anchor = (.24, 1.))
plt.show()
# plt.savefig('timing.png')
