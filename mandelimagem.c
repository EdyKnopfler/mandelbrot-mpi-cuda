#include "mandelimagem.h"

#include <math.h>
#include <stdio.h>
#include <png.h>
#include <stdlib.h>

#define CANAL_R 0
#define CANAL_G 1
#define CANAL_B 2

void gravar_imagem(REAL *imagem, int largura, int altura, char *arquivo) {
    FILE *fp = fopen(arquivo, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    REAL *row_real;
    png_bytep row_byte;
    int lin, col, tonalidade;
    
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, largura, altura,
         8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    row_byte = malloc(3 * largura * sizeof(png_byte));

    for (lin = 0, row_real = imagem; lin < altura; lin++, row_real += largura) {
        for (col = 0; col < largura; col++) {
            tonalidade = truncf(row_real[col]);
            row_byte[col*3 + CANAL_R] = tonalidade;
            row_byte[col*3 + CANAL_G] = tonalidade;
            row_byte[col*3 + CANAL_B] = tonalidade;
        }
        png_write_row(png_ptr, row_byte);
    }

    png_write_end(png_ptr, NULL);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row_byte);
    fclose(fp);
}
