import numpy as np
import matplotlib.pyplot as plt

data  = np.loadtxt("plasma_timing.dat.10000")
data2  = np.loadtxt("plasma_timing.dat.18000")
data3  = np.loadtxt("plasma_timing.dat.20000")

size_m = data[:,0]
sec1   = data[:,1]
sec2   = data2[:,1]
sec3   = data3[:,1]
rep    = 6     # number of repetition for each measure
lvec   = len(data) / rep

s  = np.zeros(lvec)
t1 = np.zeros(lvec)
t2 = np.zeros(lvec)
t3 = np.zeros(lvec)

err1 = np.zeros(lvec)
err2 = np.zeros(lvec)
err3 = np.zeros(lvec)
speedup1 = np.zeros(lvec)
speedup2 = np.zeros(lvec)
speedup3 = np.zeros(lvec)

i = 0
j = 0
count = 0
t_tmp1 = 0.
t_tmp2 = 0.
t_tmp3 = 0.

while i < len(sec1):
    while j < rep:
        t_tmp1 += sec1[i + j]
        t_tmp2 += sec2[i + j]
        t_tmp3 += sec3[i + j]

        j += 1


    s[count]  = size_m[i]

    t1[count] = t_tmp1 / rep
    t2[count] = t_tmp2 / rep
    t3[count] = t_tmp3 / rep

    t_tmp1 = 0.
    t_tmp2 = 0.
    t_tmp3 = 0.
    j -= rep

    while j < rep:
        t_tmp1 += (sec1[i + j] - t1[count])**2
        t_tmp2 += (sec2[i + j] - t2[count])**2
        t_tmp3 += (sec3[i + j] - t3[count])**2

        j += 1

    err1[count] = (t_tmp1 / (rep - 1.))**0.5
    err2[count] = (t_tmp2 / (rep - 1.))**0.5
    err3[count] = (t_tmp3 / (rep - 1.))**0.5
    
    t_tmp1 = 0.
    t_tmp2 = 0.
    t_tmp3 = 0.
    count += 1
    i += rep
    j = 0

speedup1 = t1[0] / t1
speedup2 = t2[0] / t2
speedup3 = t3[0] / t3

plt.figure()
plt.plot(s, s, label = 'Theor')
plt.errorbar(s, speedup3, yerr=err3, label = '10000x10000')
plt.errorbar(s, speedup2, yerr=err2, label = '18000x18000')
plt.errorbar(s, speedup1, yerr=err1, label = '20000x20000')

plt.title('Speedup for PLASMA library (ELCID)')
plt.xlabel('# Threads')
plt.ylabel('Speedup')
plt.legend(bbox_to_anchor = (.38, 1.))
# plt.show()
plt.savefig('scaling_plasma_elcid.png')
