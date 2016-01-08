CC=icc
C_FLAGS=-O3 -I$(PLASMA_INC)  -openmp -I${MKLROOT}/include  
PLASMA_LIBRARY= -L$(PLASMA_LIB) -lplasma -lcoreblasqw -lcoreblas -lquark -llapacke -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm  -L$(HWLOC_HOME)/lib -lhwloc -openmp 

.PHONY: all
all: test_plasma.x  

test_plasma.x: Diag_Random_plasma.o FillMatrix_plasma.o cptimer.o
	$(CC) -o test_plasma.x Diag_Random_plasma.o FillMatrix_plasma.o cptimer.o $(PLASMA_LIBRARY) 

cptimer.o: cptimer.c
	${CC} ${C_FLAGS} -c cptimer.c

FillMatrix_plasma.o: FillMatrix_plasma.c
	${CC} ${C_FLAGS} -c FillMatrix_plasma.c

Diag_Random_plasma.o: Diag_Random_plasma.c
	${CC} ${C_FLAGS} -c Diag_Random_plasma.c

clean: 
	rm -rf *~ include/*.h~
	rm -rf *.x 
	rm -rf *.o 
	rm -rf *.mod 
	rm -rf *.out
	rm -rf *.dat
