#include <iostream>
#include <mpi.h>
#include <stdlib.h> //provides atoi
#include <mkl_blacs.h>
#include <mkl_scalapack.h>
#include <fstream>

using namespace std;

extern "C" {
  void descinit_(MKL_INT *desc, MKL_INT *m, MKL_INT *n, MKL_INT *mb, MKL_INT *nb, MKL_INT *irsrc, MKL_INT *icsrc,MKL_INT *ictxt, MKL_INT *lld, MKL_INT *info);
  int  numroc_( MKL_INT *n, MKL_INT *nb, MKL_INT *iproc, MKL_INT *isrcproc, MKL_INT *nprocs);
  void pdelset_(double *A, int *i, int *j, MKL_INT *descA, double *alpha);
  
}


int main(int argc, char *argv[])
{

  MPI::Init(argc, argv); 
  int rank = MPI::COMM_WORLD.Get_rank();
  int nodes = MPI::COMM_WORLD.Get_size();
  
  if (argc<5)
    {
      if (rank == 0) cout<<" Usage: mprirun -np [nodes]  "<<argv[0]<<" [Matrix size] [CPU x Row] [CPU x Col] [BlockSize] "<<endl;	
      MPI::Finalize();
      return 0;
    }
  
  MKL_INT n = (MKL_INT)atoi(argv[1]);
  MKL_INT nprow = (MKL_INT)atoi(argv[2]);
  MKL_INT npcol = (MKL_INT)atoi(argv[3]);
  MKL_INT nb = (MKL_INT)atoi(argv[4]);
  MKL_INT numproc = nprow*npcol;
  if (rank == 0)
    {
      cout<<endl<<" Matrix size = "<<n<<endl;
      cout<<" Processor grid = "<<nprow<<" x "<<npcol<<" = "<<numproc<<endl;
      cout<<" Blocking factor = "<<nb<<endl<<endl;
    }
  
  
  /* check that numproc is equal to number of nodes */
  if (numproc != nodes)
    {
      if (rank==0) cout<<" Error, number of cpu specfied in mpirun ( "<<nodes<<" )is different from  "<<nprow<<" x "<<npcol<<" = "<<numproc<<endl;
      MPI::Finalize();
      return 0;
    }
  
	/* here goes your code */
  
  /* Hint 	initialize grid 
   *	
 *		blacs_get_       -> retrieves default system context
 *		blacs_gridinit_  -> initialize grid
 *		blacs_gridinfo_  -> sets process coordinates from context
 */
  int icontxt = 0, wh = 0;
  int my_nrow, my_ncol;
  char C[] = "R";

  blacs_get_(&wh, &wh, &icontxt);
  blacs_gridinit_(&icontxt, C, &nprow, &npcol);
  blacs_gridinfo_(&icontxt, &nprow, &npcol, &my_nrow, &my_ncol);
  
  /* Hint	get local sizes of the matrices
     MKL_INT N_loc_row = numroc_  -> NUMber of Row Or Columns. Set the local number of rows and columns for the distributed matrix
     MKL_INT N_loc_col = numroc_
  */
  
  MKL_INT N_loc_row = numroc_(&n, &nb, &my_nrow, &wh, &nprow);
  MKL_INT N_loc_col = numroc_(&n, &nb, &my_ncol, &wh, &npcol);

  /* Hint  allocate local storage */
  /*	double* A = new double[..
	double* W = new double[..
	double* Z = new double[..
  */
  double* A = new double[N_loc_row * N_loc_col];
  double* W = new double[N_loc_row * N_loc_col];
  double* Z = new double[N_loc_row * N_loc_col];
  
  MKL_INT descA[9];
  MKL_INT descZ[9];

  /* Hint 	fill matrix descriptor 
     descinit_(...
  */
  int info;
  descinit_(descA, &n, &n, &nb, &nb, &wh, &wh, &icontxt, &N_loc_row, &info);
  descinit_(descZ, &n, &n, &nb, &nb, &wh, &wh, &icontxt, &N_loc_row, &info);
  
  /* the descritopr structure is showed. DO NOT write by hand, use descinit to fill the descriptor. 
   *
   * descA[0] = 1; // descriptor type
   * descA[1] = ictxt; // blacs context
   * descA[2] = n; // global number of rows
 * descA[3] = n; // global number of columns
 * descA[4] = nb; // row block size
 * descA[5] = nb; // column block size (DEFINED EQUAL THAN ROW BLOCK SIZE)
 * descA[6] = 0; // initial process row(DEFINED 0)
 * descA[7] = 0; // initial process column (DEFINED 0)
 * descA[8] = N_loc_row; // leading dimension of local array
 */
  
  //filling the matrix, should be symmetric
  double alpha;
  for (int i=1;i<=n;i++)
    {
      for (int j=1;j<=n;j++)
	{
	  alpha=(double)(i+j);
	  pdelset_(A,&i,&j,descA,&alpha);			
	}
    }
  
  double* work;
  int lwork = -1;
  work = new double[2];
  
  //Hint  do a work space queryworkspace query
  int ione = 1;	
  char N[] = "N";
  char L[] = "L";
  
  double start, end;

  pdsyev(N, L, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info);

  lwork = (int)work[0];
  delete [] work;
  work = new double[lwork];
  
  //Hint  now call the diagonalization routine with the correct workspace size. ADD TIMING EVALUATION!!!!

  start = MPI_Wtime();
  pdsyev(N, L, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info);
  end = MPI_Wtime();

  //print output to file
  if (rank==0)
    {
      ofstream output;
      ofstream timing;
      output.open("4000new_eigenvalues.dat",ios_base::trunc);
      timing.open("4000new_timing.dat", ios_base::app);

      cout << "Printing output " << endl;
      for (int i=0;i<n;i++)
	{
	  output<<W[i] <<endl;		
	}
      output.close();

      timing << n << "\t" << end - start << endl;
      timing.close();
      // cout << "Elapsed time: " << end - start << " s" << endl;
    }
  delete [] A;
  delete [] W;
  delete [] Z;
  delete [] work;

  MPI_Barrier(MPI::COMM_WORLD);
  
  /*Hint  exit the grid */	
  blacs_gridexit_(&icontxt);
  
  MPI::Finalize();
  return 0;
}
