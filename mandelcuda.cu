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

__global__ void gpu_mandelbrot(REAL *imagem, REAL c0x, REAL c0y, REAL c1x, REAL c1y, int largura, int altura, int M) {
    REAL dx, dy, x, y;
    thrust::complex<REAL> z, c;
    int n = largura * altura;
    int indice = blockDim.x * blockIdx.x + threadIdx.x;
    int lin, col, iter;
    
    if (indice >= n) return;
    
    lin = indice / largura;
    col = indice % largura;
    dx = (c1x - c0x) / largura;
    dy = (c1y - c0y) / altura;
    
    z = thrust::complex<REAL>((REAL) 0.0, (REAL) 0.0);
    for (iter = 0; iter < M; iter++) {
        x = c0x + col*dx;
        y = c0y + lin*dy;
        c = thrust::complex<REAL>(x, y);
        z = thrust::pow(z, (REAL) 2.0) + c;
        if (thrust::abs(z) > (REAL) 2.0) break;
    }
    
    imagem[indice] = 255.0 - iter * 255.0 / M;
}

REAL *mandelbrot_cuda(int M, REAL c0x, REAL c0y, REAL c1x, REAL c1y, int largura, int altura,
                      int threads_por_bloco) {
    int n = largura * altura;
    int blocos = (n + threads_por_bloco - 1) / threads_por_bloco;
    REAL *imagem, *d_imagem;
    imagem = (REAL *) malloc(n*sizeof(REAL));
    cudaAssert(cudaSetDevice(0));
    cudaAssert(cudaMalloc(&d_imagem, n*sizeof(REAL)));
    gpu_mandelbrot<<<blocos, threads_por_bloco>>>(d_imagem, c0x, c0y, c1x, c1y, largura, altura, M);
    cudaAssert(cudaPeekAtLastError());
    cudaAssert(cudaDeviceSynchronize());
    cudaAssert(cudaMemcpy(imagem, d_imagem, n*sizeof(REAL), cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_imagem));
    return imagem;
}
