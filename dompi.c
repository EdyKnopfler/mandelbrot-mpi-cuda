#include <mpi.h>
#include <assert.h>

#include "dompi.h"
#include "mandelbrot.h"
//#include "mandelcuda.h"
#include "mandelimagem.h"

// Mandelbrot iterations
#define M 100

// This is the function to be executed by all worker_threads. It receives a task
// and a pointer where to put the partial result (the part of the image being processed).
REAL *thread_work(int start, int end, struct parameters params) {

    return mandelbrot(start, end, M, 
                      params.c0x, params.c0y, params.c1x, params.c1y, params.width, params.height);

}


void do_mpi(struct parameters params, int num_workers, int taskid) {
    REAL *the_big_picture = NULL;
    REAL* my_portion = NULL;

    int err = 0;
    int N = params.width * params.height;
    int work_size = N / num_workers;
    int start, end;
        
    // If all threads do this calculation, we can save the scatter step!
    if (N % num_workers > 0) {
        N = work_size * num_workers + work_size;
        work_size = N / num_workers;
    }
    
    // My work
    start = taskid * work_size;
    end = (taskid + 1) * work_size;
    printf("Process %d: %d to %d\n", taskid, start, end);
    my_portion = thread_work(start, end, params);

    // Send the result back to the root process (in this case, 0)
    if (taskid == 0)
        the_big_picture = (REAL*) malloc(N * sizeof(REAL));
    err |= MPI_Gather(my_portion,  // Buffer to send
            work_size,             // Number of elements
            MPI_REAL_MACRO,        // ...
            the_big_picture,       // Buffer to recv (root process only)
            work_size,             // Number of elements for each recv
            MPI_REAL_MACRO,
            0,                     // root
            MPI_COMM_WORLD);
            
    if (err)
        DIE("There was an MPI communication error.\n");

    if (taskid == 0)
        gravar_imagem(the_big_picture, params.width, params.height, params.output);
}
