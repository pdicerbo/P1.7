P1.7 Results
============================

In this report I will present the results of the benchmarks done during the P1.7 course.
In the first part I show the tests done with different libraries (Intel MKL and OpenBLAS) and the performances obtained from three different functions.
Then I switch into the test of the DSYEV function on different machines to test scaling properties of the ScaLAPACK, PLASMA and MAGMA libraries.

Intel MKL vs OpenBLAS
#######################

The purpose of this section is to compare the performances obtained from two different implementations of three particular functions. The functions are the following:

**DAXPY**, that perform the following operation:

.. math::

   y = a * x + y;

where :math:`a` is a moltiplicative constant while :math:`x` and :math:`y` are vectors of dimension N. This function perform :math:`2 N` operations.

**DGEMV**, that perform the following operation:

.. math::

   y = a * A * x + b * y;

where :math:`a` and :math:`b` are two moltiplicative constants while :math:`x` and :math:`y` are vectors of dimension N. This function perform :math:`2 N^2` operations.

**DGEMM**, that perform the following operation:

.. math::

   C = a * A * B + b * C;

where :math:`a` and :math:`b` are two moltiplicative constants while :math:`A`, :math:`B` and :math:`C` are vectors of dimension N. This function perform :math:`2 N^3` operations.

This functions are choosen because of their different scaling properties with respect to the size of the problem. In the following there is the results obtained on the ULISSE cluster:

.. image:: plots/mkl_vs_openblas.png
   :scale: 20

As you can see, the plot clearly show that DAXPY and DGEMV are memory bounded while DGEMM is cpu bounded.
	   


