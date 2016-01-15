CC=mpicxx

# MKLROOT variable for elcid
# MKLROOT=/opt/intel/mkl

# MKLROOT variable for cosint
MKLROOT=/opt/cluster/intel/composer_xe_2015.0.090/mkl

LIBS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm

all=8x3pdsyev.x

8x3pdsyev.x: pdsyev.o
	$(CC) -m64 -I${MKLROOT}/include $^ $(LIBS) -o $@

%.o: %.cc
	$(CC) -m64 -I${MKLROOT}/include -c $^

clean:
	rm -rf *.o *.x
