import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing.dat")

size_m = data[:,0]
sec    = data[:,1]

rep = 6     # number of repetition for each matrix size
lvec = len(data) / rep

s = np.zeros(lvec)
t = np.zeros(lvec)

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
    count += 1
    i += rep
    j = 0

plt.figure()
plt.plot(s, t)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.show()
# plt.savefig('timing.png')
