#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include "mkl.h"
#include "mkl_blas.h"

double cclock()
{
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

int main(int argc, char** argv){
  
  if(argc != 2){
    printf("\n\tUsage:\n\t%s matrix_size\n\n", argv[0]);
    return 0;
  }

  double *A, *B, *C;
  int N = atoi(argv[1]);
  int iter = 10;
  int i, j, k;
  double start, end;
  FILE* fp;

  A = (double*)malloc(N * N * sizeof(double));
  B = (double*)malloc(N * N * sizeof(double));
  C = (double*)malloc(N * N * sizeof(double));

  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++){
      A[i*N + j] = (double) (i*N + j);
      B[i*N + j] = (double) (i*N - j);
    }

  if(N > 2000){
    start = cclock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1., A, N, B, N, 0., C, N);
    end = cclock();
  }
  else{
    start = cclock();
    for(j = 0; j < iter; j++)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1., A, N, B, N, 0, C, N);
    end = cclock();    
  }

#ifdef _DEBUG
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      printf("\t%lg", C[i*N + j]);
    }
    printf("\n");
  }

  printf("\n\t%lg\n", end - start);
  printf("\tGFLOPS: %lg \n\n", 2.*N*N*N/1.e9/(end - start));

#endif /* _DEBUG */

  
  fp = fopen("dgemm_mkl.dat", "a+");
  if(N > 2000)
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N*N*N/1.e9/(end - start));
  else
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N*N*N*iter/1.e9/(end - start));
  fclose(fp);

  /* **** DGEMV SECTION **** */
  double* vec = (double*)malloc(N * sizeof(double));
  double* res = (double*)malloc(N * sizeof(double));
  for(j = 0; j < N; j++)
    vec[j] = (double) j;

  if(N > 2000){
    start = cclock();
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1., A, N, vec, 1., 0., res, 1.);
    end = cclock();
  }
  else{
    start = cclock();
    for(j = 0; j < iter; j++)
      cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1., A, N, vec, 1., 0., res, 1.);
    end = cclock();    
  }
  fp = fopen("dgemv_mkl.dat", "a+");
  if(N > 2000)
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N*N/1.e9/(end - start));
  else
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N*N*iter/1.e9/(end - start));
  fclose(fp);  
  /* **** DAXPY SECTION **** */
  
  if(N > 2000){
    start = cclock();
    cblas_daxpy(N, 2., vec, 1, res, 1);
    end = cclock();
  }
  else{
    start = cclock();
    for(j = 0; j < iter; j++)
      cblas_daxpy(N, 2., vec, 1, res, 1);
    end = cclock();    
  }
  
  fp = fopen("daxpy_mkl.dat", "a+");
  if(N > 2000)
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N/1.e9/(end - start));
  else
    fprintf(fp, "%d\t%lg\t%lg\n", N, end - start, 2.*N*iter/1.e9/(end - start));
  fclose(fp);  

  free(A);
  free(B);
  free(C);
  free(vec);
  free(res);

  return 0;
}
