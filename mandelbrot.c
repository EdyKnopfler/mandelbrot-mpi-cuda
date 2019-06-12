#include "mandelbrot.h"
#include "mandelimagem.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

REAL *mandelbrot_omp(int start, int end, int M, REAL c0x, REAL c0y, REAL c1x, REAL c1y, int largura, int altura) {
    REAL dx, dy, x, y;
    REAL complex z, c;
    int indice, lin, col, iter, N = largura * altura;
    REAL *imagem;

    imagem = malloc((end - start) * sizeof(REAL));
    dx = (c1x - c0x) / largura;
    dy = (c1y - c0y) / altura;
    
    for (indice = start; indice < end; indice++) {
        if (indice >= N) break;  // Lidando com a sobra
        
        lin = indice / largura;
        col = indice % largura;
        
        z = 0.0 + 0.0 * I;
        x = c0x + col*dx;
        y = c0y + lin*dy;
        c = x + y * I;
        
        for (iter = 0; iter < M; iter++) {
            z = cpowf(z, 2.0) + c;
            if (cabsf(z) > 2.0) break;
        }
        
        imagem[indice - start] = 255.0 - iter * 255.0 / M;
    }
    
    return imagem;
}
