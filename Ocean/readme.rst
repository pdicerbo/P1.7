Info
=========================

The Version3 of the code works only if :math:`(xdim - 2) % threads == 0`, where :math:`threads = blockDim.x * gridDim.x`. Actually the serial version of the code
runs faster than the gpu version with the standard configuration (:math:`xdim = 4098; ydim = 4098; gridDim.x = 16; blockDim.x = 128; timesteps = 100`).
This depends both by the small size of the matrix and the small amount of running threads, even if obtain good performances with this configuration is not easy.
Actually, I think that it would be possible improuving performances adopting a 2D grid, but I don't have enough time to implement this version of the kernel.
