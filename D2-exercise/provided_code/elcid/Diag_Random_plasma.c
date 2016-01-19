/* 
 * Template program to compute eigenvalues of an Hermitian matrix using 
 * the PLASMA_dsyev subroutine 
 * int PLASMA_dsyev 	( 	PLASMA_enum 	 	jobz,
 * 				PLASMA_enum  		uplo,
 * 				int  			N,
 *			  	double		 	*A,
 * 				int  			LDA,
 * 				double 			*W,
 * 				PLASMA_desc 		*descT,
 * 				double		 	*Q,
 * 				int  			LDQ 
 * 			) 	
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
#include <stdio.h>
#include "utils_plasma.h"

int main( int argc, char *argv[] ){

  int n = 0;
  int nthreads;
  FILE* fp;
  FILE* timing;
  double start, end;

  // Hint - Declare variables and  pointers for PLASMA_dsyev
  PLASMA_desc *descT;
  PLASMA_enum job = PlasmaNoVec;
  PLASMA_enum up  = PlasmaUpper;

  // Minimal instructions for reading the matrix size from standard input
  if( argc != 3 ){
     printf("\n");
     printf("%-20s \n\n", "Sample Plasma_dsyev code.") ;
     printf("%-8s %-4s %-20s", "usage:", argv[0], "{matrix size} {num_threads}");
     return 0; 
  }

  n = atoi(argv[1]);
  nthreads = atoi(argv[2]);

  // Hint - Allocate arrays and fill variables relevant to PLASMA_dsyev
  double *A, *w, *vec;
  A = (double*) malloc(n * n * sizeof(double));
  w = (double*) malloc(n * sizeof(double));
  
  // Initialization of the sample matrix
  FillMatrix_plasma(A, n);

  // Print the Initliazed Hermitian matrix
  fp = fopen( "matrix.dat", "w+b" );
  fwrite( A, sizeof(PLASMA_Double_t), n*n, fp );
  fclose(fp);
   
  // Hint - Initialize Plasma to 0 so PLASMA_NUM_THREADS environment variable is used. ( Function PLASMA_Init(int) )
  PLASMA_Init(nthreads);

  // Hint - Allocate workspace for the diagonalization call ( PLASMA_Alloc_Workspace_dsyev(int, int, PLASMA_desc*)  )
  PLASMA_Alloc_Workspace_dsyev(n, n, &descT);

  // Time measuring for the eingenvalues computation
  start = cclock(); 	
  // Hint - call PLASMA_dsyev
  PLASMA_dsyev(job, up, n, A, n, w, descT, vec, n);

  end =  cclock();

  fprintf( stdout, "\nTime to compute eigenvalues for an Hermitian matrix of dimension %d using LAPACK function LAPACKE_dsyev: %.3g (seconds)\n", n, end - start );

  // Write the eigenvalues of the Hermitian matrix
  fp = fopen( "eigenvalues.dat", "w+b" );
  fwrite( w, sizeof(PLASMA_Double_t), n, fp );
  fclose(fp);
  
  timing = fopen("plasma_timing.dat", "a+");
  fprintf(timing, "%d\t%lg\n", nthreads, end - start);
  fclose(timing);

  // Hint - Deallocate arrays 
  free(A);
  free(w);
 // Hint - Finalize PLASMA (PLASMA_Finalize() )
  PLASMA_Finalize();

  return 0;
}
