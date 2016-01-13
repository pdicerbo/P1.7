#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include "mkl.h"

int mnk=4;

double mytime(){
  timeval v;
  gettimeofday(&v,0);
  return v.tv_sec+v.tv_usec/1000000.0;
}

void matrixmul_mnk(double* c,double* a,double* b){
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
	      mnk, mnk, mnk, 1, a, mnk, b, mnk, 1, c, mnk);
}

void my_matrix_mul(double* c, double* a, double* b){
  int i, j, k;
  int i_tmp;

  for(i = 0; i < mnk; i++){
    i_tmp = i * mnk;
#pragma unroll_and_jam(2)
    for(j = 0; j < mnk; j++){

      c[i_tmp + j] = 0.;

      //#pragma simd vectorlengthfor(double)
      for(k = 0; k < mnk; k++){
	c[i_tmp + j] += a[i_tmp + k] * b[k*mnk + j];
      }
    }
  }
}

int main(void){
  int iter=10;
  int nmatrices=10000000;
  int size=mnk*mnk*nmatrices;
  double* a= (double*) _mm_malloc(sizeof(double)*size,64);
  double* b= (double*) _mm_malloc(sizeof(double)*size,64);
  double* c= (double*) _mm_malloc(sizeof(double)*size,64);
  double time1,time2,time3;
  for(int i=0;i<size;i++){
    a[i]=rand();
    b[i]=rand();
    c[i]=rand();
  }
  
  time1=mytime();
  for(int n=0;n<iter;n++){
    for(int i=0;i<size;i+=mnk*mnk){
      matrixmul_mnk(&c[i],&a[i],&b[i]);
	  //you code goes here	
      //matrixmul_mnk_opt1(&c[i],&a[i],&b[i]);
    }
  }
  time2=mytime();

  for(int n=0;n<iter;n++)
    for(int i=0;i<size;i+=mnk*mnk)
      my_matrix_mul(&c[i],&a[i],&b[i]);

  time3=mytime();


  printf("\n\ntime dgemm= %f s\n", time2-time1);
  printf("perf = %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);

  printf("\ntime my_matrix_mul= %f s\n", time3-time2);
  printf("perf = %f GFLOPs\n\n\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time3-time2)/1000.0/1000.0/1000.0);
}
