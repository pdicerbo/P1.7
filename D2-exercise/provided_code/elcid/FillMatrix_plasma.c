/* 
 * Routine to generate a random Hermitian matrix for using PLASMA (C interface)
 *
 * Code created for hands-on based evaluation for the course
 * "P1.7 - Advanced Computer Architectures and Optimizations" 
 * for the Master in High Performance Computing @SISSA/ICTP
 *
 * Developed by: G.P. Brandino (brandino@sissa.it) and M. Atambo (matambo@ictp.it) 
 * Reviewd by: I. Girotto ( igirotto@ictp.it )
 *
 */

#include <stdlib.h>
#include <math.h>
#include "utils_plasma.h"

void FillMatrix_plasma( PLASMA_Double_t *A, int n ){

  double temp, temp1;
  int i,j ;

  srand( time(NULL) );

 /* Even though the PLASMA interface is a C interface, 
  *  the matrix is read in col maj order
  */

  for ( i = 0; i < n; i++ ){
    for ( j = 0; j <= i; j++ ){
      
      if( i == j ){
             
	temp = rand() / ( (double) RAND_MAX );
	A[ j * n + i ] = temp;
      }
      else{
	
	temp = rand() / ( (double) RAND_MAX );
	A[ j * n + i ] = temp;
	A[ j * n + i ] = temp;
      }
    }
  }
}
