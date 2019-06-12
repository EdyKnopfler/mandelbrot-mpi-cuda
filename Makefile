CC = gcc
NVCC = nvcc
REAL ?= double
MPI_REAL_MACRO ?= MPI_DOUBLE
CFLAGS=-Wall -Wextra -std=c99 -pedantic -O3 -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -DREAL=$(REAL) -DMPI_REAL_MACRO=$(MPI_REAL_MACRO)
#LDFLAGS = -lm -lmpi -lpng -Xcompiler
LDFLAGS = -lm -lmpi -lpng -fopenmp
CFLAGS_CUDA = -g -G -DREAL=$(REAL) -DMPI_REAL_MACRO=$(MPI_REAL_MACRO)


dmbrot: dmbrot.o dompi.o mandelbrot.o mandelimagem.o

mandelcuda.o: mandelcuda.cu
	$(NVCC) $(CFLAGS_CUDA) -c -o $@ $^

.PHONY: clean
clean:
	rm -f dmbrot *.o

