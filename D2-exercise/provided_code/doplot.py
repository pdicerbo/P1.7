import numpy as np
import matplotlib.pyplot as plt

data  = np.loadtxt("plasma_timing.safe")
ndata = np.loadtxt("scaling_timing.dat")

size_m = data[:,0]
sec    = data[:,1]
nsec   = ndata[:,1]
rep    = 6     # number of repetition for each matrix size
lvec   = len(data) / rep

s  = np.zeros(lvec)
t  = np.zeros(lvec)
nt = np.zeros(lvec)

err = np.zeros(lvec)
nerr = np.zeros(lvec)
speedup  = np.zeros(lvec)
nspeedup = np.zeros(lvec)

i = 0
j = 0
count = 0
t_tmp = 0.
n_tmp = 0.

while i < len(sec):
    while j < rep:
        t_tmp += sec[i + j]
        n_tmp += nsec[i + j - rep]
        j += 1


    s[count] = size_m[i]
    t[count] = t_tmp / rep
    nt[count - 1] = n_tmp / rep
    
    t_tmp = 0.
    n_tmp = 0.
    j -= rep

    while j < rep:
        t_tmp += (sec[i + j] - t[count])**2
        n_tmp += (nsec[i + j - rep] - nt[count - 1])**2
        j += 1

    err[count] = (t_tmp / (rep - 1.))**0.5
    nerr[count - 1] = (n_tmp / (rep - 1.))**0.5
    
    t_tmp = 0.
    n_tmp = 0.
    count += 1
    i += rep
    j = 0

speedup  = t[0] / t
nspeedup = nt[0] / nt

plt.figure()
plt.errorbar(s, speedup, yerr=err, label = 'PLASMA')
plt.errorbar(s, nspeedup, yerr=nerr, label = 'ScaLAPACK')
plt.plot(s, s, label = 'Teor')
plt.title('Speedup for matrix size = 8000')
plt.xlabel('# Threads')
plt.ylabel('Speedup')
plt.legend(bbox_to_anchor = (.35, 1.))
# plt.show()
plt.savefig('scaling.png')
