P1.7 Results
============================

In this report I will present the results of the benchmarks done during the P1.7 course.
In the first part I show the tests done with different libraries (Intel MKL and OpenBLAS) and the performances obtained from three different functions.
Then I switch into the test of the DSYEV function on different machines to test scaling properties of the ScaLAPACK, PLASMA and MAGMA libraries.

Most of the plot that will be shown were obtained by repeating 6 (otherwise 10) times the same executable in order to have a bit of statistics.
Error bars associated with each point was obtained by calculating the standard deviation of these measures.

Intel MKL vs OpenBLAS
#######################

The purpose of this section is to compare the performances obtained from two different implementations of the following three particular functions:

**DAXPY**, that perform the following operation:

.. math::

   y = a * x + y;

where :math:`a` is a moltiplicative constant while :math:`x` and :math:`y` are vectors of dimension N. This function perform :math:`2 N` floating point operations.

**DGEMV**, that perform the following operation:

.. math::

   y = a * A * x + b * y;

where :math:`a` and :math:`b` are two moltiplicative constants while :math:`x` and :math:`y` are vectors of dimension N. This function perform :math:`2 N^2` floating point operations.

**DGEMM**, that perform the following operation:

.. math::

   C = a * A * B + b * C;

where :math:`a` and :math:`b` are two moltiplicative constants while :math:`A`, :math:`B` and :math:`C` are vectors of dimension N. This function perform :math:`2 N^3` floating point operations.

This functions are choosen because of their different scaling properties with respect to the size of the problem. In the following there is the results obtained on the ULISSE cluster:

.. image:: plots/mkl_vs_openblas.png
   :scale: 20

Both sources were compiled with gcc compiler. As you can see, the plot clearly shows that increasing the size of the matrix DAXPY and DGEMV are memory bounded while DGEMM is compute bounded.
For DAXPY and DGEMV and for matrix size less than 2500 the Intel MKL library achieves better performances with respect to OpenBLAS, while for size
greater than 2500 the performances are more or less the same. This appens also for DGEMM for matrix size greater than 1000, while for matrix size less than 1000 seems that ObenBLAS achieves slightly
better performances with respect to Intel MKL.
	   

SCALAPACK DSYEV Benchmark
##########################



