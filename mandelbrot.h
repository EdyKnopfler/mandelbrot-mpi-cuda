#ifndef _MANDELBROT

#define _MANDELBROT

REAL *mandelbrot_omp(int start, int end, int M, REAL c0x, REAL c0y, REAL c1x, REAL c1y, int largura, int altura);

#endif
