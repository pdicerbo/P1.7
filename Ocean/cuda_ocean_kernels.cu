


#ifdef VERSION1
__global__ void ocean_kernel(int *grid, int xdim, int ydim, int offset)
{
    int threads = gridDim.x*blockDim.x;
    int threadId  = blockDim.x*blockIdx.x + threadIdx.x;

    if (threads > (xdim-2)*(ydim-2)) {
        threads = (xdim-2)*(ydim-2);
        if (threadId >= threads) {
            return;
        }
    }

    int chunk = (xdim-2)*(ydim-2)/threads;
    int start = threadId * chunk;
    int end = (threadId + 1) * chunk;

    int threadsPerRow = (xdim - 2);

    for (int i=start; i<end; i++) {
        if (offset) {
            if (i%2) continue;
        } else {
            if (!(i%2)) continue;
        }

        int row = i / threadsPerRow;
        int col = i % threadsPerRow;

        int loc = xdim + row * xdim + col;
        if (offset) {
            loc += (row%2) ? 1 : 0;
            loc += 1;
        } else {
            loc += (row%2) ? 0 : 1;
        }
        // printf("Row: %d, Col: %d\n", row, col);
        // printf("loc: %d\n", loc);

        grid[loc] = (grid[loc]
                  + grid[loc - xdim]
                  + grid[loc + xdim]
                  + grid[loc + 1]
                  + grid[loc - 1])
                  / 5;
    }
}
#endif


#ifdef VERSION2
__global__ void ocean_kernel(int *grid, int xdim, int ydim, int offset)
{
    int threads = gridDim.x*blockDim.x;
    int threadId  = blockDim.x*blockIdx.x + threadIdx.x;

    if (threads > (xdim-2)*(ydim-2)) {
        threads = (xdim-2)*(ydim-2);
        if (threadId >= threads) {
            return;
        }
    }

    int chunk = (xdim-2)*(ydim-2)/threads;
    int start = 0.;

    int threadsPerRow = (xdim - 2);

    for (int i=start; i < chunk; i++) {
      if (offset){
            if (threadIdx.x % 2) continue;
        } else {
            if (!(threadIdx.x % 2)) continue;
        }

        int row = (i * threads) / threadsPerRow;
        int col = (i * threads) % threadsPerRow;

        int loc = xdim + row * xdim + col + threadId;
        if (offset) {
            loc += (row%2) ? 1 : 0;
            loc += 1;
        } else {
            loc += (row%2) ? 0 : 1;
        }

        grid[loc] = (grid[loc]
                  + grid[loc - xdim]
                  + grid[loc + xdim]
                  + grid[loc + 1]
                  + grid[loc - 1])
                  / 5;
    }
}
#endif


#ifdef VERSION3

__global__ void split_array_kernel(int *grid, int *red_grid, int *black_grid, int xdim, int ydim)
{
  // This kernel should take the contents of grid and copy all of the red
  // elements into red_grid and all of the black elements into black_grid

  int threads  = gridDim.x * blockDim.x;
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (threads > (xdim - 2)*(ydim - 2) ) {
    threads = (xdim - 2) * (ydim - 2);
    if (threadId >= threads) {
      return;
    }
  }
  
  int chunk = (xdim - 2) * (ydim - 2) / threads;
  if( (xdim - 2) * (ydim - 2) % threads )
    return;

  int threadsPerRow = (xdim - 2);
  int red;

  for (int i = 0; i < chunk; i++) {
    int row = (i * threads + threadId) / threadsPerRow;
    int col = (i * threads + threadId) % threadsPerRow;
    int loc = (row + 1) * xdim + 1 + col;

    if(row % 2){
      if(col % 2)
	red = 0;
      else
	red = 1;
    }
    else{
      if(col % 2)
	red = 1;
      else
	red = 0;
    }

    if( red )
      red_grid[loc / 2] = grid[loc];
    else
      black_grid[loc / 2] = grid[loc];

    // BOUNDARIES
    // first row
    if(row == 0){
      if( red )
    	black_grid[col / 2 + 1] = grid[loc - xdim];
      else
    	red_grid[col / 2] = grid[loc - xdim];
    }
    // last row
    if(row == ydim - 3){
      if( red )
    	black_grid[(loc + xdim) / 2 ] = grid[loc + xdim];
      else
    	red_grid[(loc + xdim) / 2] = grid[loc + xdim];
    }

    // left column
    if(col == 0){
      if( red )
    	black_grid[loc / 2] = grid[loc - 1];
      else
    	red_grid[loc / 2] = grid[loc - 1];
    }

    // right column
    if(col == xdim - 3){
      if( red )
    	black_grid[loc / 2] = grid[loc + 1];
      else
    	red_grid[loc / 2] = grid[loc + 1];
    }
  }
}

__global__ void unsplit_array_kernel(int *grid, int *red_grid, int *black_grid, int xdim, int ydim)
{
  // "Inverse" of the previous function
  int threads  = gridDim.x * blockDim.x;
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (threads > (xdim - 2)*(ydim - 2) ) {
    threads = (xdim - 2) * (ydim - 2);
    if (threadId >= threads) {
      return;
    }
  }
  
  int chunk = (xdim - 2) * (ydim - 2) / threads;
  if( (xdim - 2) * (ydim - 2) % threads )
    return;

  int threadsPerRow = (xdim - 2);
  int red;

  for (int i = 0; i < chunk; i++) {
    int row = (i * threads + threadId) / threadsPerRow;
    int col = (i * threads + threadId) % threadsPerRow;
    int loc = (row + 1) * xdim + 1 + col;

    if(row % 2){
      if(col % 2)
	red = 0;
      else
	red = 1;
    }
    else{
      if(col % 2)
	red = 1;
      else
	red = 0;
    }

    if( red )
      grid[loc] = red_grid[loc / 2];
    else
      grid[loc] = black_grid[loc / 2];

    // BOUNDARIES
    if(row == 0){
      if( red )
    	grid[loc - xdim] = black_grid[col / 2 + 1];
      else
    	grid[loc - xdim] = red_grid[col / 2];
    }
    // last row
    if(row == ydim - 3){
      if( red )
    	grid[loc + xdim] = black_grid[(loc + xdim) / 2 ];
      else
    	grid[loc + xdim] = red_grid[(loc + xdim) / 2];
    }

    // left column
    if(col == 0){
      if( red )
    	grid[loc - 1] = black_grid[loc / 2];
      else
    	grid[loc - 1] = red_grid[loc / 2];
    }

    // right column
    if(col == xdim - 3){
      if( red )
    	grid[loc + 1] = black_grid[loc / 2];
      else
    	grid[loc + 1] = red_grid[loc / 2];
    }
  }
}

__global__ void ocean_kernel(int *red_grid, int *black_grid, int xdim, int ydim, int offset)
{
  int threads  = gridDim.x * blockDim.x;
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if (threads > ((xdim - 2) * (ydim - 2) / 2)) {
    threads = (xdim - 2) * (ydim - 2) / 2;
    if (threadId >= threads)
      return;
  }

  if( (xdim - 2) * (ydim - 2) % (2 * threads) )
    return;

  int chunk = (xdim - 2) * (ydim - 2) / (2 * threads);
  int threadsPerRow = (xdim - 2) / 2;
  int i, edge, left, right;
  int shift = xdim / 2;

  for(i = 0; i < chunk; i++){

    int row = (i * threads + threadId) / threadsPerRow;
    int loc = threadId + i * threads + shift;

    if(offset){
      //black_grid update
      edge = (row + 1) / 2;
      edge *= 2;
      loc  += edge; // need to skip the elements in the boundaries
      if( row & 1 ){
	left = loc - 1;
	right = loc;
      }
      else{
	left = loc;
	right = loc + 1;
      }

      black_grid[loc] = (black_grid[loc]
			 + red_grid[loc - shift]
			 + red_grid[loc + shift]
			 + red_grid[left]
			 + red_grid[right]) / 5;
    }
    else{
      // red_grid update
      loc += 1;
      edge = row / 2;
      edge *= 2;
      loc  += edge; // need to skip the elements in the boundaries

      if( row & 1 ){
	left = loc;
	right = loc + 1;
      }
      else{
	left = loc - 1;
	right = loc;
      }

      red_grid[loc] = (red_grid[loc]
		       + black_grid[loc - shift]
		       + black_grid[loc + shift]
		       + black_grid[left]
		       + black_grid[right]) / 5;
    }

  }
}

#endif // VERSION3