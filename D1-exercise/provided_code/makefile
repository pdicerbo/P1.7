CC=mpicxx

# MKLROOT variable for elcid
# MKLROOT=/opt/intel/mkl

# LIBS=-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lpthread -lm

# compile with the following flag for use static iMPI
#LIBS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lm

# compile with the following flag for use static OpenMPI
LIBS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm

all=pdsyev.x

5x4pdsyev.x: pdsyev.o
	$(CC) -I${MKLROOT}/include $^ $(LIBS) -o $@

%.o: %.cc
	$(CC) -I${MKLROOT}/include -c $^

clean:
	rm -rf *.o *.x
