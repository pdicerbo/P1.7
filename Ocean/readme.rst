Info
=========================

The Version3 of the code works only if $(xdim - 2) % threads == 0$, where $threads = blockDim.x * gridDim.x$.
