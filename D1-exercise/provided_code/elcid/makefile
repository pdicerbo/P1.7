CC=mpicxx

# MKLROOT variable for elcid
MKLROOT=/opt/intel/mkl

LIBS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm

INCL=-m64 -I${MKLROOT}/include

all=10pdsyev.x

10pdsyev.x: pdsyev.o
	$(CC) $(INCL) $^ $(LIBS) -o $@

%.o: %.cc
	$(CC) $(INCL) -c $^

clean:
	rm -rf *.o *.x
