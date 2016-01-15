import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing.dat")

size_m = data[:,0]
sec    = data[:,1]

rep = 6     # number of repetition for each matrix size
lvec = len(data) / rep

s = np.zeros(lvec)
t = np.zeros(lvec)
err = np.zeros(lvec)

i = 0
j = 0
count = 0
t_tmp = 0.

while i < len(sec):
    while j < rep:
        t_tmp += sec[i + j]
        j += 1


    s[count] = size_m[i]
    t[count] = t_tmp / rep

    t_tmp = 0.
    j -= rep

    while j < rep:
        t_tmp += (sec[i + j] - t[count])**2
        j += 1

    err[count] = (t_tmp / (rep - 1.))**0.5

    t_tmp = 0.
    count += 1
    i += rep
    j = 0

print(t)
print(err)
print(len(t))
print(len(err))
plt.figure()
# plt.plot(s, t)
plt.errorbar(s, t, yerr=err)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.show()
# plt.savefig('timing.png')
