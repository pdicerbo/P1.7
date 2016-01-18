import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("daxpy_mkl.dat")
data2 = np.loadtxt("dgemv_mkl.dat")
data3 = np.loadtxt("dgemm_mkl.dat")

size_m = data1[:,0]
sec1   = data1[:,2]
sec2   = data2[:,2]
sec3   = data3[:,2]

rep = 10     # number of repetition for each matrix size
lvec = len(data1) / rep

s = np.zeros(lvec)
t1 = np.zeros(lvec)
err1 = np.zeros(lvec)
t2 = np.zeros(lvec)
err2 = np.zeros(lvec)
t3 = np.zeros(lvec)
err3 = np.zeros(lvec)

i = 0
j = 0
count  = 0
t_tmp1 = 0.
t_tmp2 = 0.
t_tmp3 = 0.

while i < len(sec1):
    while j < rep:
        t_tmp1 += sec1[i + j]
        t_tmp2 += sec2[i + j]
        t_tmp3 += sec3[i + j]
        j += 1


    s[count] = size_m[i]
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


mkl_t1 = np.copy(t1)
mkl_t2 = np.copy(t2)
mkl_t3 = np.copy(t3)
mkl_err1 = np.copy(err1)
mkl_err2 = np.copy(err2)
mkl_err3 = np.copy(err3)
# REPEAT FOR OPENBLAS DATA
data1 = np.loadtxt("daxpy_openblas.dat")
data2 = np.loadtxt("dgemv_openblas.dat")
data3 = np.loadtxt("dgemm_openblas.dat")

size_m = data1[:,0]
sec1   = data1[:,2]
sec2   = data2[:,2]
sec3   = data3[:,2]

rep = 10     # number of repetition for each matrix size
lvec = len(data1) / rep

s = np.zeros(lvec)
t1 = np.zeros(lvec)
err1 = np.zeros(lvec)
t2 = np.zeros(lvec)
err2 = np.zeros(lvec)
t3 = np.zeros(lvec)
err3 = np.zeros(lvec)

i = 0
j = 0
count  = 0
t_tmp1 = 0.
t_tmp2 = 0.
t_tmp3 = 0.

while i < len(sec1):
    while j < rep:
        t_tmp1 += sec1[i + j]
        t_tmp2 += sec2[i + j]
        t_tmp3 += sec3[i + j]
        j += 1


    s[count] = size_m[i]
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


plt.figure()
# plt.plot(s, t)
plt.errorbar(s, t3, yerr=err3, label='dgemm_mkl')
plt.errorbar(s, mkl_t3, yerr=mkl_err3, label='dgemm_openblas')
plt.errorbar(s, t2, yerr=err2, label='dgemv_mkl')
plt.errorbar(s, mkl_t2, yerr=mkl_err2, label='dgemv_openblas')
plt.errorbar(s, t1, yerr=err1, label='daxpy_mkl')
plt.errorbar(s, mkl_t1, yerr=mkl_err1, label='daxpy_openblas')
plt.title('Performance for DGEMM, DGEMV and DAXPY (ULISSE)')
plt.xlabel('Matrix Size')
plt.ylabel('GFLOPS')
plt.legend(bbox_to_anchor = (1., .85))
# plt.show()
plt.savefig('scaling_library.png')
