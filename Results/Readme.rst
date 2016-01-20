P1.7 Results
============================

In this report I will present the results of the benchmarks done during the P1.7 course.
In the first part I show the tests done with different libraries (Intel MKL and OpenBLAS) and the performances obtained from three different functions.
Then I switch into the test of the DSYEV function on different machines to test scaling properties of the ScaLAPACK, PLASMA and MAGMA libraries.

Intel MKL vs OpenBLAS
#######################

The purpose of this section is to compare the performances obtained from two different implementations of three particular functions. The functions are the following:

DAXPY, that perform the following operation (vector - vector):

.. html:: latex

   y = a \\cdot x + y; a \\in \\mathbb{R}; x, y vectors; \\mathbf{2 * n operations}

DGEMV, that perform the following operation (matrix - vector):

.. code::

   $y = \alpha * A * x + \beta * y; \alpha , \beta \in \mathbb{R}$; $x, y$ vectors; $A$ matrix; $\mathbf{2 * n^2 operations}$


DGEMM, that perform the following operation (matrix - matrix):

.. code::

   $C = \alpha * A * B + \beta * C; \alpha, \beta \in \mathbb{R}; A, B, C$ matrices; $\mathbf{2 * n^3 operations}$

As you can see, this functions are choosen because of their different scaling properties with respect to the size of the problem.


