CC=gcc
CFLAGS=
LIBS=-Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
INCL=-I$(MKLROOT)/include

all=blas_scaling.x

blas_scaling.x: blas_scaling.o
	$(CC) $(LIBS) $^ -o $@

%.o:%.c
	$(CC) -O2 $(INCL) -c $^

clean:
	rm -rf *.o *.x