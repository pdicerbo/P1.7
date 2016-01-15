CXX= gcc
CXXFLAGS= -O3 -std=c99
LDFLAGS =
CUDACC = nvcc
CUFLAGS = -O3 -arch=sm_35

all: serial_ocean cuda_ocean

serial_ocean: serial_ocean.c ocean_main.o
	$(CXX) $(CXXFLAGS) ocean_main.o serial_ocean.c -o serial_ocean $(LDFLAGS)

cuda_ocean: ocean_main.o cuda_ocean.cu cuda_ocean_kernels.cu
	$(CUDACC) $(CUFLAGS) ocean_main.o cuda_ocean.cu -o cuda_ocean $(LDFLAGS)

ocean_main.o: ocean_main.c
	$(CXX) $(CXXFLAGS) ocean_main.c -c -o ocean_main.o

clean:
	rm -f serial_ocean
	rm -f cuda_ocean
	rm -f ocean_main.o
