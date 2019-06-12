#include <mpi.h>
#include <assert.h>

#include "dompi.h"
#include "mandelbrot.h"
//#include "mandelcuda.h"
#include "mandelimagem.h"

// Mandelbrot iterations
#define M 100

void declare_task_type(MPI_Datatype *type) {
    int err = 0;
    MPI_Datatype internal_types[] = {MPI_INT, MPI_INT};
    int types_len[] = {1, 1};
    int num_elements = STATIC_LEN(internal_types);
    MPI_Aint offsets[] = {
        offsetof(struct task, start),
        offsetof(struct task, end)
    };

    assert(offsets[num_elements - 1] + sizeof(int) == sizeof(struct task));

    err |= MPI_Type_create_struct(num_elements,
            types_len,
            offsets,
            internal_types,
            type
    );
    err |= MPI_Type_commit(type);
    if (err)
        DIE("There was a MPI type registration failure.\n");
}

// This is the function to be executed by all worker_threads. It receives a task
// and a pointer where to put the partial result (the part of the image being processed).
REAL *thread_work(struct task this_task, struct parameters params) {

    printf("%d %d %f %f %f %f %d %d %s %s\n", this_task.start, this_task.end, params.c0x, params.c0y, 
            params.c1x, params.c1y, params.width, params.height,
           params.processor, params.output);


    return mandelbrot(this_task.start, this_task.end, M, 
                      params.c0x, params.c0y, params.c1x, params.c1y, params.width, params.height);

}


void do_mpi(struct parameters params, int num_workers, int taskid) {
    MPI_Datatype MPI_STRUCT_TASK;
    
    struct task* tasks = NULL;
    REAL *the_big_picture = NULL;

    struct task task;
    REAL* my_portion = NULL;

    int err = 0, N = params.width * params.height, work_size = 0;

    declare_task_type(&MPI_STRUCT_TASK);

    if (taskid == 0)
    {
        work_size = N / num_workers;
        
        if (N % num_workers > 0) {
            N = work_size * num_workers + work_size;
            work_size = N / num_workers;
        }
        
        tasks = (struct task*) malloc(num_workers * sizeof(struct task));
        the_big_picture = (REAL*) malloc(N * sizeof(REAL));

        for (int i = 0; i < num_workers; ++i)
        {
            tasks[i].start = i * work_size;
            tasks[i].end = (i + 1) * work_size;
        }
    }

    // Scatter will distribute the work equally between all processes,
    // including myself
    err |= MPI_Scatter(tasks,       // Buffer to send
            1,               // Number of elements sent to each process
            MPI_STRUCT_TASK, // Our custom datatype
            &task,           // Buffer to recv data.
            1,               // Number of elements of this buffer
            MPI_STRUCT_TASK, // ...
            0,               // Root
            MPI_COMM_WORLD   // ...
    );

    printf("Process %d recv: %d, %d\n", taskid, task.start, task.end);
    my_portion = thread_work(task, params);

    // Send the result back to the root process (in this case, 0)
    err |= MPI_Gather(my_portion,  // Buffer to send
            task.end - task.start,             // Number of elements
            MPI_REAL_MACRO,        // ...
            the_big_picture,       // Buffer to recv (root process only)
            work_size, // Number of elements for each recv
            MPI_REAL_MACRO,
            0,                     // root
            MPI_COMM_WORLD);
            
    if (err)
        DIE("There was an MPI communication error.\n");

    if (taskid == 0)
    {
        gravar_imagem(the_big_picture, params.width, params.height, params.output);
    }
}
