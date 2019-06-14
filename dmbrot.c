#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "dompi.h"

int main(int argc, char **argv) {
    int world_size, taskid;
    int err = 0;
    struct parameters params;

    // Send attributes to all processes, and initialize MPI.
    err |= MPI_Init(&argc, &argv);

    // Get how many processes there are in the world MPI_COMM_WORLD
    err |= MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Which process am I?
    err |= MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    
    if (err)
        DIE("There was an MPI initialization error.\n");

    // Argument parsing
    if (argc < 10) {
        printf("usage: %s <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <OUTPUT>\n",
               argv[0]);
        return 1;
    }
    
    params.c0x = atof(argv[1]);
    params.c0y = atof(argv[2]);
    params.c1x = atof(argv[3]);
    params.c1y = atof(argv[4]);
    params.width = atoi(argv[5]);
    params.height = atoi(argv[6]);
    params.processor = argv[7];
    params.threads = atoi(argv[8]);
    params.output = argv[9];

    do_mpi(params, world_size, taskid);

    MPI_Finalize();
    return 0;
}

