#ifndef _MANDELCUDA

#define _MANDELCUDA

#ifdef __cplusplus
extern "C" {
#endif

REAL *mandelbrot_cuda(int M, REAL c0x, REAL c0y, REAL c1x, REAL c1y, int largura, int altura,
                      int threads_por_bloco);

#ifdef __cplusplus
}
#endif

#endif
