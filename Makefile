CC = gcc
NVCC = nvcc
REAL ?= double
MPI_REAL_MACRO ?= MPI_DOUBLE
CFLAGS=-Wall -Wextra -std=c99 -pedantic -O3 -fopenmp -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -DREAL=$(REAL) -DMPI_REAL_MACRO=$(MPI_REAL_MACRO)
LDFLAGS = -lm -lmpi -lpng -Xcompiler -fopenmp
CFLAGS_CUDA = -Xptxas --opt-level=3 -DREAL=$(REAL)

dmbrot: dmbrot.o dompi.o mandelbrot.o mandelcuda.o mandelimagem.o
	$(NVCC) -o $@ $^ $(LDFLAGS)

mandelcuda.o: mandelcuda.cu
	$(NVCC) $(CFLAGS_CUDA) -c -o $@ $^

.PHONY: clean
clean:
	rm -f dmbrot *.o

