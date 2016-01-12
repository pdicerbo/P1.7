#include <stdio.h>
#include <stdlib.h>
#include "immintrin.h"
#include <xmmintrin.h>
#include <sys/time.h>
#include <time.h>

double mytime(){
  timeval v;
  gettimeofday(&v,0);
  return v.tv_sec+v.tv_usec/1000000.0;
}



int main(){
  
  int size=4*20000 + 2; //100002;
  int iter=100002;
  int t, i;
  double* f1= (double*) _mm_malloc(sizeof(double)*size,32);
  double* f2= (double*) _mm_malloc(sizeof(double)*size,32);
  double* tmpptr;
  double a,b;
  double c;
  double dx,dt;
  double time1,time2;
  
  // set some meaningfull parameters
  c=0.1;
  dx=0.01;
  dt=0.01;

  // precompute some values
  a=c*dt/dx;
  a=a*a;
  b=2*(1-a);
  
  // initialize to zero
  for(int i=0;i<size;i++){
    f1[i]=0;
    f2[i]=0;
  }
  // make some delta peaks
  f1[size/2]=0.1;
  f2[size/2]=-0.1;
  
  time1 = mytime();

  __m256d v, w, z;

  for(t = 0; t < iter; t++){
    for(i = 1; i < size - 1; i += 4){
      
      v = _mm256_loadu_pd(&f1[i + 1]);
      w = _mm256_loadu_pd(&f1[i - 1]);

      v = _mm256_add_pd(v, w); // f1[i+1]+f1[i-1]
      
      w = _mm256_set1_pd(a);
      v = _mm256_mul_pd(v, w); // a*(f1[i+1]+f1[i-1])

      w = _mm256_loadu_pd(&f2[i]);
      v = _mm256_sub_pd(v, w); // a*(f1[i+1]+f1[i-1])-f2[i]

      w = _mm256_loadu_pd(&f1[i]);
      z = _mm256_set1_pd(b);  
      w = _mm256_mul_pd(w, z); // b*f1[i]

      v = _mm256_add_pd(v, w); // a*(f1[i+1]+f1[i-1])+b*f1[i]-f2[i]
      
      _mm256_storeu_pd(&f2[i], v);
    }
    tmpptr=f1;
    f1=f2;
    f2=tmpptr;
  }

  time2 = mytime();

  printf("\n\ttime used: %lg\n\n", time2 - time1);

  return 0;
}
