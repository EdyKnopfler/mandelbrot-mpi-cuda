#include <stdio.h>
#include <stdlib.h>
#include <thrust/complex.h>

#include "mandelcuda.h"

void cudaAssert(cudaError_t erro) {
    if (erro != cudaSuccess) {
        printf("Erro código: %d\n", erro);
        abort();
    }
}

__global__ void gpu_mandelbrot(REAL *imagem, int n, int start, int M, struct parameters params) {
    REAL dx, dy, x, y;
    thrust::complex<REAL> z, c;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int lin, col, iter;
    
    if (idx >= n) return;
    
    lin = (idx + start) / params.width;
    col = (idx + start) % params.width;
    dx = (c1x - c0x) / params.width;
    dy = (c1y - c0y) / params.height;
    
    z = thrust::complex<REAL>((REAL) 0.0, (REAL) 0.0);
    x = c0x + col*dx;
    y = c0y + lin*dy;
    c = thrust::complex<REAL>(x, y);
    
    for (iter = 0; iter < M; iter++) {
        z = thrust::pow(z, (REAL) 2.0) + c;
        if (thrust::abs(z) > (REAL) 2.0) break;
    }
    
    imagem[idx] = 255.0 - iter * 255.0 / M;
}

REAL *mandelbrot_cuda(int start, int end, int M, struct parameters params) {
    int n = end - start;
    int blocos = (n + params.threads - 1) / params.threads;
    REAL *imagem, *d_imagem;
    imagem = (REAL *) malloc(n*sizeof(REAL));
    cudaAssert(cudaSetDevice(0));
    cudaAssert(cudaMalloc(&d_imagem, n*sizeof(REAL)));
    gpu_mandelbrot<<<blocos, params.threads>>>(d_imagem, n, start, M, params);
    cudaAssert(cudaPeekAtLastError());
    cudaAssert(cudaDeviceSynchronize());
    cudaAssert(cudaMemcpy(imagem, d_imagem, n*sizeof(REAL), cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_imagem));
    return imagem;
}
