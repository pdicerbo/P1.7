D2 - exercise 
=============

1. Using the template code Diag_Random_plasma.c, write a code to call the PLASMA_dsyev_ routine, which calculates eigenvalues and optionally eigenvectors or an Hermitian matrix.

   In short you should

     - Allocate all the relevant arrays 
     - Initialize PLASMA to 0 (such that it uses the number of threads given by the environmental variable PLASMA_NUM_THREADS)
     - Allocate the workspace (using PLASMA_Alloc_Workspace_dsyev(int, int, PLASMA_desc*)
     - Call PLASMA_dsyev
     - Deallocate arrays
     - Call PLASMA_Finalize

You can find the official PLASMA documentation here_. 

2 On a single node, compare the perfomance with SCALAPACK, for the same number of computing units.

.. _PLASMA_dsyev: http://icl.cs.utk.edu/projectsfiles/plasma/html/doxygen/group__double_gac7ea19b1441c1325f45c0f6a9cfd8a8a.html 
.. _here: http://icl.cs.utk.edu/projectsfiles/plasma/html/doxygen/

