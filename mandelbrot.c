#include "mandelbrot.h"

#include <omp.h>
#include <stdlib.h>
#include <complex.h>

REAL *mandelbrot(int start, int end, int M, struct parameters params) {
    REAL dx, dy, x, y;
    REAL complex z, c;
    int idx, lin, col, iter, N = params.width * params.height;
    REAL *imagem;

    imagem = malloc((end - start) * sizeof(REAL));
    dx = (params.c1x - params.c0x) / params.width;
    dy = (params.c1y - params.c0y) / params.height;
    
    omp_set_num_threads(params.threads);
    
    #pragma omp parallel for
    for (idx = start; idx < end; idx++) {
        if (idx >= N) break;  // Lidando com a sobra
        
        lin = idx / largura;
        col = idx % largura;
        
        z = 0.0 + 0.0 * I;
        x = c0x + col*dx;
        y = c0y + lin*dy;
        c = x + y * I;
        
        for (iter = 0; iter < M; iter++) {
            z = cpowf(z, 2.0) + c;
            if (cabsf(z) > 2.0) break;
        }
        
        imagem[idx - start] = 255.0 - iter * 255.0 / M;
    }
    
    return imagem;
}
