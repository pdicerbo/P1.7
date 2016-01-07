D1 - exercise
=============

1 Implement a parallel dsyev ( Double SYmmetric EigenValues)
  The scalapack routine to use is called "pdsyev_".
  Use the mkl implementation of scalapack on argo.
  Please refer to the official documentation_ 

  Follow the direction in the provided_code in pdsyev.cc

  In short, you should:

  - initialize the blacs grid
  - calculate local storage and allocate it
  - initialize the distributed array descriptor
  - make a workspace query
  - call the diagonalization routine


.. _pdsyev : https://software.intel.com/en-us/node/470034?wapkw=pdsyev 

.. _documentation : https://software.intel.com/en-us/node/521455

