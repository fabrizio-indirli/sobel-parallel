#include "datastr.h"
#include <stdlib.h>
#include <math.h>

__global__ void kernel_sobel_filter(pixel* sobel, pixel* pi, int height, int width);


#ifdef __cplusplus
extern "C"{
#endif 
    void apply_sobel_filter(int width, int height, pixel * pi);

    void apply_sobel_filter_omp(int width, int height, pixel * pi);

    void apply_sobel_filter_part(int width, int height, pixel * pi, int startheight, int finalheight);

    void sobel_filter_auto(int width, int height, pixel * pi);
#ifdef __cplusplus
}
#endif