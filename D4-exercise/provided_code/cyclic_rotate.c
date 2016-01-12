#include <stdio.h>
#include <stdlib.h>
#include "immintrin.h"

int main(){

  int dim = 4;
  int i;
  double* vec = (double*)malloc(dim * sizeof(double));
  double* rec = (double*)malloc(dim * sizeof(double));

  for(i = 0; i < dim; i++)
    vec[i] = (double) i;

  __m256d v;

  v = _mm256_loadu_pd(vec);

  v = _mm256_permute_pd(v, 5);

  v = _mm256_permute2f128_pd(v, v, 1);

  _mm256_storeu_pd(rec, v);

  /* printing data */
  printf("\n\tdata sent\n");
  for(i = 0; i < dim; i++)
    printf("\t%lg", vec[i]);

  printf("\n\n\tdata received\n");
  for(i = 0; i < dim; i++)
    printf("\t%lg", rec[i]);

  printf("\n");

  return 0;
}
