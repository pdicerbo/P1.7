/* template code to run magma_dsyevd(_gpu, _m)
 * 
 *   Created by G.P. Brandino for the course 
 *   "P1.7 Advanced Computer Architectures and Optimizations"
 *   for the Master in High Performance Computing @SISSA/ICTP 
 *
 *	none 	magma_dsyevd	hybrid CPU/GPU routine where the matrix is initially in CPU host memory.
 *      _gpu    magma_dsyevd_gpu	hybrid CPU/GPU routine where the matrix is initially in GPU device memory.
 *	_m 	magma_dsyevd_m	hybrid CPU/multiple-GPU routine where the matrix is initially in CPU host memory.
 *
 */

#include <stdio.h>
#include <magma.h>



int main (int argc, char* argv[])
{
  
  real_Double_t   gpu_time, cpu_time, mcpu_time;
  int i,j;
  
  if (argc!=2)
    {
      printf("\n Sample Magma dsyevd(_gpu)  code. \n");  
      printf(" Usage: %s {matrix size} \n", argv[0]);
      return 0; 
    }
  
  printf("\n Sample Magma dsyevd(_gpu,_m)  code. \n"); 
  printf(" Matrix size ---> %d  -->  %f MB \n\n",atoi(argv[1]), atoi(argv[1])*atoi(argv[1])*sizeof(double)/1048576.0);
  
  FILE *first, *sec, *third;
  
  // HINT - Initialize magma (Function  magma_init() ) 
  magma_init();
  // Now working on magma_dsyevd 
  printf(" Calling magma_dsyevd ...\n");
  
  // Declare all the  variables relevant to magma_dsyevd
  magma_vec_t jobz = MagmaNoVec;
  magma_uplo_t uplo = MagmaUpper;
  magma_int_t n = atoi(argv[1]);
  int lda = n;
  magma_int_t lwork, liwork, info;
  magma_int_t *iwork;
  
  double *A, *w, *work;
  
  /* 
     magma_int_t magma_dsyevd	(	magma_vec_t jobz, 
                                        magma_uplo_t uplo, 
					magma_int_t n,
					double *a, 
					magma_int_t lda, 
					double *w,
					double *work, 
					magma_int_t lwork,
					magma_int_t *iwork, 
					magma_int_t liwork, 
					magma_int_t *info
					)
					
  */
  
  lwork=-1;
  liwork=-1;

  // HINT - Make a workspace query, using function magma_dsyevd, to set the correct values to lwrk and liwork 
  work = (double*)malloc(sizeof(double));
  iwork = (int*)malloc(sizeof(int));
  magma_dsyevd(jobz, uplo, n, A, n, w, work, lwork, iwork, liwork, &info);
  
  lwork = (magma_int_t) work[0];
  liwork = iwork[0];
  free(work);
  free(iwork);
  work = (double*)malloc(lwork * sizeof(double));
  iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t));

  // HINT - Allocate space for the arrays A, w, iwork, work. 
  // magma_dsyevd uses only arrays allocated on the cpu memory.
  // You can use either 
  // 	magma_?malloc_cpu(void**, magma_int_t)
  //      magma_?malloc_pinned(void**, magma_int_t)

  A = (double*)malloc(n * n * sizeof(double));
  w = (double*)malloc(n * sizeof(double));
  
  for(i=0; i < n; ++i )
    {
      for(j=0; j < n; ++j )
	{
	  A[i*lda+j]=(double)(i*j);
	}
    }	 
  
  
  cpu_time = magma_wtime();
  // HINT - Call magma_dsyevd 
  magma_dsyevd(jobz, uplo, n, A, n, w, work, lwork, iwork, liwork, &info);
  
  cpu_time = magma_wtime() - cpu_time;
  
  if (info!=0)
    printf(" Error %d \n", info);
  printf(" Done with magma_dsyevd! It took %f seconds \n\n", cpu_time);

  first = fopen("dsyevd.dat", "a+");
  fprintf(first, "%d\t%lg\n", n, cpu_time);
  fclose(first);

  // HINT - Free allocated memory, using either 
  //	magma_free_cpu(void*)
  //	magma_free_pinned(void*)
  
  //--------------------------------------------------------------------------------------------
  //Now working on magma_dsyevd_gpu 
  printf(" Calling magma_dsyev_gpu ...\n");	
  
  //HINT - Declare all the variables relevant to magma_dsyevd_gpu
  /*
    magma_int_t magma_dsyevd_gpu  ( 	magma_vec_t jobz, 
                                        magma_uplo_t uplo,
					magma_int_t n,
					double *da, 
					magma_int_t ldda,
					double *w,
					double *wa,  
					magma_int_t ldwa,
					double *work, 
					magma_int_t lwork,
					magma_int_t *iwork, 
					magma_int_t liwork,
					magma_int_t *info
					)
  */
  
  lwork  = -1;
  liwork = -1;
  
  // HINT - Make a workspace query, using function magma_dsyevd_gpu, to set the correct values to lwrk and liwork 
  double *da, *wa;

  magma_dsyevd_gpu(jobz, uplo, n, da, n, w, wa, n, work, lwork, iwork, liwork, &info);

  lwork  = work[0];
  liwork = iwork[0];

  free(work);
  free(iwork);
  
  /* work = (double*)malloc(lwork * sizeof(double)); */
  /* iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t)); */

  magma_malloc_cpu((void**) &wa, n * n * sizeof(double));
  magma_malloc_cpu((void**) &work, lwork * sizeof(double));
  magma_malloc_cpu((void**) &iwork, liwork * sizeof(magma_int_t));

    // HINT - Allocate space for the arrays A, dA,  w, iwork, work. 
    // magma_dsyevd_gpu uses requires w, iwork and work to be allocated on the host, while dA is allocated on the device. You need also an array A on the host, which is filled and then transfered to the device by means of magma_dsetmatrix 
    // You can use either 
    //      magma_?malloc_cpu(void**, magma_int_t)
    //      magma_?malloc_pinned(void**, magma_int_t)
    // for the host allocation and 
    // 	magma_?alloc(void**, magma_int_t) 
    // for the device allocation
    
  magma_dmalloc(&da, n * n);

  for(i=0; i < n; ++i )
    {
      for(j=0; j < n; ++j )
	{
	  wa[i*lda+j]=(double)(i*j);
	}
    }
  
  
  // HINT - Use function magma_dsetmatrix to transfer the array A on the host to the array dA on the device

  magma_dsetmatrix(n, n, wa, n, da, n);

  gpu_time = magma_wtime();
  // HINT - Call magma_dsyevd_gpu

  magma_dsyevd_gpu(jobz, uplo, n, da, n, w, wa, n, work, lwork, iwork, liwork, &info);
  
  gpu_time = magma_wtime() - gpu_time;
  
  if (info!=0)
    printf(" Error %d \n", info);
  printf(" Done with magma_dsyevd_gpu! It took %f seconds \n\n", gpu_time);

  sec = fopen("dsyevd_gpu.dat", "a+");
  fprintf(sec, "%d\t%lg\n", n, gpu_time);
  fclose(sec);

  
  // HINT - Free allocated memory, using either 
  //      magma_free_cpu(void*)
  //      magma_free_pinned(void*)
  // for host deallocation and 
  // 	magma_free(void*)
  // for device deallocation
  
  
  //--------------------------------------------------------------------------------------------
  // Now working on magma_dsyevd_m
  
  printf("Calling magma_dsyevd_m ...\n");
  
  /*
    magma_int_t magma_dsyevd_m      (       magma_int_t     ngpu,
                                            magma_vec_t     jobz,
  					    magma_uplo_t    uplo,
  					    magma_int_t     n,
  					    double *        A,
  					    magma_int_t     lda,
  					    double *        w,
  					    double *        work,
  					    magma_int_t     lwork,
  					    magma_int_t *   iwork,
  					    magma_int_t     liwork,
  					    magma_int_t *   info
  					    )
  */
  
  // Declare all the variables relevant to magma_dsyevd_gpu
  
  lwork  = -1;
  liwork = -1;
  
  // HINT - Make a workspace query, using function magma_dsyevd_m, to set the correct values to lwrk and liwork
  magma_dsyevd_m(2, jobz, uplo, n, A, n, w, work, lwork, iwork, liwork, &info);
  
  lwork  = work[0];
  liwork = iwork[0];
    
  free(work);
  free(iwork);
  
  work = (double*)malloc(lwork * sizeof(double));
  iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t));

  // HINT - Allocate space for the arrays A, w, iwork, work.
  // magma_dsyevd_m uses only arrays allocated on the cpu memory.
  // You can use either
  //      magma_?malloc_cpu(void**, magma_int_t)
  //      magma_?malloc_pinned(void**, magma_int_t)
  
  for(i=0; i < n; ++i )
    {
      for(j=0; j < n; ++j )
  	{
  	  A[i*lda+j]=(double)(i*j);
  	}
    }
  
  
  mcpu_time = magma_wtime();
  //HINT - Call magma_dsyevd_m
  magma_dsyevd_m(2, jobz, uplo, n, A, n, w, work, lwork, iwork, liwork, &info);
  
  mcpu_time = magma_wtime() - mcpu_time;
  
  if (info!=0)
    printf(" Error %d \n", info);
  printf(" Done with magma_dsyevd_m! It took %f seconds \n\n", mcpu_time);

  third = fopen("dsyevd_m.dat", "a+");
  fprintf(third, "%d\t%lg\n", n, mcpu_time);
  fclose(third);

  // HINT - Free allocated memory, using either 
  //      magma_free_cpu(void*)
  //      magma_free_pinned(void*)
  
  // HINT - Finalize magma using function magma_finalize() 
  free(A);
  free(w);
  free(work);
  free(iwork);
  free(da);
  free(wa);

  magma_finalize();
  
  return 0;
}
