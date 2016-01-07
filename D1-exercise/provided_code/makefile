CC=mpicxx
# LIBS=-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lpthread -lm

LIBS=${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lm

all=pdsyev.x

pdsyev.x: pdsyev.o
	$(CC) -I${MKLROOT}/include $^ $(LIBS) -o $@

%.o: %.cc
	$(CC) -c $^

clean:
	rm -rf *.o *.x
