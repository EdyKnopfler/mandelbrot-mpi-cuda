#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "mandelbrot.h"
//#include "mandelcuda.h"
#include "mandelimagem.h"


struct task {
    int start, end;
};

// Get the length of an statically allocated array. This do not work for
// malloc'ed arrays.
#define STATIC_LEN(x) (sizeof(x)/sizeof(*x))

#define DIE(...) { \
    fprintf(stderr, __VA_ARGS__); \
    exit(EXIT_FAILURE); \
}

MPI_Datatype MPI_STRUCT_TASK;

void declare_task_type(MPI_Datatype *type)
{
    int err = 0;

    // Type of each struct member
    MPI_Datatype internal_types[] = {MPI_INT, MPI_INT};

    // How many elements there are per member (is it a statically allocated array)?
    int types_len[] = {1, 1};

    // Just get how many elements there are in the struct.
    int num_elements = STATIC_LEN(internal_types);

    // Get offsets of each elements
    MPI_Aint offsets[] = {
        offsetof(struct task, start),
        offsetof(struct task, end)
    };

    // Ensure that the calculated size is equal to the type calculated by the compiler.
    assert(offsets[num_elements - 1] + sizeof(int) == sizeof(struct task));

    // Create the MPI typename
    err |= MPI_Type_create_struct(num_elements,
            types_len,
            offsets,
            internal_types,
            type
    );

    // Commit type. Now we can use it.
    err |= MPI_Type_commit(type);

    if (err)
        DIE("There was a MPI type registration failure.\n");
}

// Command-line parameters
REAL c0x, c0y, c1x, c1y;
int width, height;
char *processor, *output;

// Mandelbrot iterations
#define M 100

// These tags are used to identify the messages. TCP does not guarantee that
// two sent messages sent will be received in the correct order.
#define TAG_TASK    0
#define TAG_RESULT  1

// This is the function to be executed by all worker_threads. It receives a task
// and a pointer where to put the partial result (the part of the image being processed).
REAL *thread_work(struct task this_task) {

    return mandelbrot_omp(this_task.start, this_task.end, M, c0x, c0y, c1x, c1y, width, height);

}

static void do_mpi(int num_workers, int taskid)
{
    struct task* tasks = NULL;
    REAL *the_big_picture = NULL;

    struct task task;
    REAL* my_portion = NULL;

    int err = 0, N = width * height, work_size = 0;

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
    my_portion = thread_work(task);

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
        gravar_imagem(the_big_picture, width, height, output);
    }
}

int main(int argc, char **argv)
{
    int world_size, taskid;
    int err = 0;

    // Send attributes to all processes, and initialize MPI.
    err |= MPI_Init(&argc, &argv);

    // Get how many processes there are in the world MPI_COMM_WORLD
    err |= MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Which process am I?
    err |= MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    
    if (err)
        DIE("There was an MPI initialization error.\n");

    // Argument parsing
    if (argc < 9) {
        printf("usage: %s <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <OUTPUT>\n",
               argv[0]);
        return 1;
    }
    
    c0x = atof(argv[1]);
    c0y = atof(argv[2]);
    c1x = atof(argv[3]);
    c1y = atof(argv[4]);
    width = atoi(argv[5]);
    height = atoi(argv[6]);
    processor = argv[7];
    output = argv[8];

    do_mpi(world_size, taskid);

    MPI_Finalize();
    return 0;
}

