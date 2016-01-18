#include <stdio.h>

//#define VERSION1
//#define VERSION15
//#define VERSION2
#define VERSION3
// #define DBG

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

    int *d_red_grid, *d_black_grid, *d_new_grid;
    int // *host_red, *host_black, 
      *mygrid, *new_grid;

    cudaMalloc(&d_new_grid, sizeof(int)*xdim*ydim);
    Check_CUDA_Error("malloc d_new_grid failed");
    cudaMalloc(&d_red_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc red_grid failed");
    cudaMalloc(&d_black_grid, sizeof(int)*xdim*ydim / 2);
    Check_CUDA_Error("malloc black_grid failed");

    split_array_kernel<<<16,512>>>(d_grid, d_red_grid, d_black_grid, xdim, ydim);
    Check_CUDA_Error("split_array_kernel launch failed");

    dim3 gridDim(4,1,1);
    dim3 blockDim(4,1,1);
    // dim3 gridDim(16,1,1);
    // dim3 blockDim(128,1,1);
    
    ocean_kernel<<<gridDim, blockDim>>>(d_red_grid, d_black_grid, xdim, ydim, 0);
    Check_CUDA_Error("unsplit_array_kernel launch failed 0");
    ocean_kernel<<<gridDim, blockDim>>>(d_red_grid, d_black_grid, xdim, ydim, 1);
    Check_CUDA_Error("unsplit_array_kernel launch failed 1");

    ocean_kernel_V2<<<gridDim, blockDim>>>(d_grid, xdim, ydim, 0);
    ocean_kernel_V2<<<gridDim, blockDim>>>(d_grid, xdim, ydim, 1);

    unsplit_array_kernel<<<16,512>>>(d_new_grid, d_red_grid, d_black_grid, xdim, ydim);
    Check_CUDA_Error("unsplit_array_kernel launch failed");

    // host_red   = (int*)malloc(xdim * ydim * sizeof(int)/2);
    // host_black = (int*)malloc(xdim * ydim * sizeof(int)/2);
    mygrid   = (int*)malloc(xdim * ydim * sizeof(int));
    new_grid = (int*)malloc(xdim * ydim * sizeof(int));
    
    // cudaMemcpy(host_red, d_red_grid, xdim*ydim*sizeof(int)/2, cudaMemcpyDeviceToHost);
    // Check_CUDA_Error("memcpy grid back failed 0");
    // cudaMemcpy(host_black, d_black_grid, xdim*ydim*sizeof(int)/2, cudaMemcpyDeviceToHost);
    // Check_CUDA_Error("memcpy grid back failed 1");
    cudaMemcpy(mygrid, d_grid, xdim*ydim*sizeof(int), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed 2");
    cudaMemcpy(new_grid, d_new_grid, xdim*ydim*sizeof(int), cudaMemcpyDeviceToHost);
    Check_CUDA_Error("memcpy grid back failed 3");

    int count = 0;
    int err_count = 0;
    int index;

    for(int j = 0; j < xdim; j++){
      for(int k = 0; k < ydim; k++){
    	index = k + j * xdim;
    	if(new_grid[index] != mygrid[index]){
    	  err_count++;
    	  // printf("%d\t", index);
    	}
    	count++;
      }
    }
    printf("\n\tnew_grid\n");
    for(int j = 0; j < xdim; j++){
      for(int k = 0; k < ydim; k++){
	printf("\t%d", new_grid[k + j * xdim]);
      }
      printf("\n");
    }
    printf("\n\tV2_grid\n");
    for(int j = 0; j < xdim; j++){
      for(int k = 0; k < ydim; k++){
	printf("\t%d", mygrid[k + j * xdim]);
      }
      printf("\n");
    }

    printf("\n\tdiff\n");
    for(int j = 1; j < xdim - 1; j++){
      for(int k = 1; k < ydim - 1; k++){
	printf("\t%d", abs(mygrid[k + j * xdim] - new_grid[k + j * xdim]));
      }
      printf("\n");
    }

    printf("\n\tErrors: %d; Correct = %d; Elements counted: %d\n", 
	   err_count, count - err_count, count);

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

    dim3 gridDim(16*4,1,1);
    dim3 blockDim(128*4,1,1);


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
