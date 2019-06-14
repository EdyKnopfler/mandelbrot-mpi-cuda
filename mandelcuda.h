#ifndef _MANDELCUDA

#define _MANDELCUDA

#ifdef __cplusplus
extern "C" {
#endif

#include "dompi.h"

REAL *mandelbrot_cuda(int start, int end, int M, struct params params);

#ifdef __cplusplus
}
#endif

#endif
