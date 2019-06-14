#ifndef _DOMPI

#define _DOMPI

#include <stdio.h>
#include <stdlib.h>

// Get the length of an statically allocated array. This do not work for
// malloc'ed arrays.
#define STATIC_LEN(x) (sizeof(x)/sizeof(*x))

// Error checking
#define DIE(...) { \
    fprintf(stderr, __VA_ARGS__); \
    exit(EXIT_FAILURE); \
}

// These tags are used to identify the messages. TCP does not guarantee that
// two sent messages sent will be received in the correct order.
//#define TAG_TASK    0
//#define TAG_RESULT  1

// Command-line parameters
struct parameters {
    REAL c0x, c0y, c1x, c1y;
    int width, height, threads;
    char *processor, *output;
};

// Generate a mandelbrot image in a distributed manner
void do_mpi(struct parameters params, int num_workers, int taskid);

#endif
