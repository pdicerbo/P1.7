Info
=========================

The Version3 of the code works only if :math:`(xdim - 2) % threads == 0`, where :math:`threads = blockDim.x * gridDim.x`.
