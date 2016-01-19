CC=icc
CFLAGS= -Wall -openmp -I$(MKLROOT)/include -I$(MAGMA_INC) -DMAGMA_WITH_MKL -DMAGMA_SETAFFINITY -DADD_
LIBS= -L$(MAGMA_LIB) -lmagma -L$(CUDADIR)/lib64 -lcublas -lcudart  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm 

all: magma_dsyevd.x

magma_dsyevd.x: magma_dsyevd.c
	$(CC) $^ $(CFLAGS) $(LIBS) -o $@

clean:
	rm *.x 
