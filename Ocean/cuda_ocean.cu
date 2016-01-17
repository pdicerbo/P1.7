#include <stdio.h>

//#define VERSION1
//#define VERSION15
//#define VERSION2
#define VERSION3
#define DBG

#include "cuda_ocean_kernels.cu"

void Check_CUDA_Error(const char *message)
{
   cudaError_t error = cudaGetLastError();
   if(error!=cudaSuccess) {
      fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
   }
}

extern "C"
void ocean (int **grid, int xdim, int ydim, int timesteps)
{
    /********************* the red-black algortihm (start)************************/
    /*
    In odd timesteps, calculate indeces with - and in even timesteps, calculate indeces with *
    See the example of 6x6 matrix, A represents the corner elements.
        A A A A A A
        A - * - * A
        A * - * - A
        A - * - * A
        A * - * - A
        A A A A A A
    */

    // Set up the GPU datastructure used in the first 3 kernel verisons

    int *d_grid;

    cudaMalloc(&d_grid, sizeof(int)*xdim*ydim);
    Check_CUDA_Error("malloc grid failed");

    cudaMemcpy(d_grid, &grid[0][0], xdim*ydim*sizeof(int), cudaMemcpyHostToDevice);
    Check_CUDA_Error("memcpy grid to device failed");

#ifdef DBG

    int *d_red_grid, *d_black_grid;
    int *host_red, *host_black, *mygrid;
    cudaMalloc(&d_red_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc red_grid failed");
    cudaMalloc(&d_black_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc black_grid failed");

    split_array_kernel<<<16,512>>>(d_grid, d_red_grid, d_black_grid, xdim, ydim);
    // split_array_kernel<<<16,128>>>(d_grid, d_red_grid, d_black_grid, xdim, ydim);
    Check_CUDA_Error("split_array_kernel launch failed");

    host_red   = (int*)malloc(xdim * ydim * sizeof(int)/2);
    host_black = (int*)malloc(xdim * ydim * sizeof(int)/2);
    mygrid     = (int*)malloc(xdim * ydim * sizeof(int));

    cudaMemcpy(host_red, d_red_grid, xdim*ydim*sizeof(int)/2, cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed 0");
    cudaMemcpy(host_black, d_black_grid, xdim*ydim*sizeof(int)/2, cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed 1");
    cudaMemcpy(mygrid, d_grid, xdim*ydim*sizeof(int), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed 2");

    int count = 0;
    int err_count = 0;
    int zero_black = 0;
    int zero_red = 0;
    int zero_count = 0;
    int index;
    for(int j = 1; j < xdim-1; j++){
      for(int k = 1; k < ydim-1; k++){
	index = k + j * xdim;
	if(count % 2){
	  if(host_red[index / 2] != mygrid[index])
	    err_count++;
	  if(host_red[index / 2] == 0)
	    ++zero_red;
	}
	else{
	  if(host_black[index / 2] != mygrid[index])
	    err_count++;
	  if(host_black[index / 2] == 0)
	    zero_black++;
	}
	count++;
      }
    }

    printf("\n\tErrors: %d; Correct = %d\n", err_count, xdim*ydim - err_count);
    printf("\tMyGrid zero count: %d\n", zero_count);
    printf("\tColor zero black: %d\n", zero_black);
    printf("\tColor zero red: %d\n", zero_red);
    printf("\tgrid[2] == %d == %d == black[1];\n", mygrid[2], host_black[1]);
    printf("\tgrid[1] == %d == %d == red[0];\n", mygrid[1], host_red[0]);
    printf("\tgrid[xdim + 1] == %d == %d == black[xdim/2];\n", mygrid[xdim+1], host_black[xdim/2]);
    printf("\tgrid[xdim + 2] == %d == %d == red[xdim/2 + 1];\n\n", mygrid[xdim+2], host_red[xdim/2 + 1]);
    exit(0);
#endif // DBG

    #if defined(VERSION3)

    // set up the GPU datastructure for the other kernel versions
    int *red_grid, *black_grid;
    cudaMalloc(&red_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc red_grid failed");
    cudaMalloc(&black_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc black_grid failed");

    split_array_kernel<<<16,512>>>(d_grid, red_grid, black_grid, xdim, ydim);
    Check_CUDA_Error("split_array_kernel launch failed");

    #endif

    dim3 gridDim(16,1,1);
    dim3 blockDim(128,1,1);


    for (int ts=0; ts<timesteps; ts++) {
        #if defined(VERSION3)
        ocean_kernel<<<gridDim, blockDim>>>(red_grid, black_grid, xdim, ydim, ts%2);
        #else
    	ocean_kernel<<<gridDim, blockDim>>>(d_grid, xdim, ydim, ts%2);
        #endif
    	Check_CUDA_Error("ocean_kernel launch failed");
    }

    #if defined(VERSION3)
    unsplit_array_kernel<<<16,512>>>(d_grid, red_grid, black_grid, xdim, ydim);
    Check_CUDA_Error("unsplit_array_kernel launch failed");
    #endif

    cudaMemcpy(&grid[0][0], d_grid, xdim*ydim*sizeof(int), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed");

    cudaFree(d_grid);


    /////////////////////// the red-black algortihm (end) ///////////////////////////
}
