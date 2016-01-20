CC=icc
CFLAGS=-I${MKLROOT}/include -O3
LIBS=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm

all=exercise1-matrixmult-blas.x

exercise1-matrixmult-blas.x: exercise1-matrixmult-blas.o
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@

exercise1-matrixmult-blas.o: exercise1-matrixmult-blas.cpp
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o *.x