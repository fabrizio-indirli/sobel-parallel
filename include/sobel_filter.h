#include "datastr.h"
#include <stdlib.h>
#include <math.h>

__global__ void compute_sobel_filter(pixel* sobel, pixel* pi, int height, int width);

#ifdef __cplusplus
extern "C"{
#endif 
    void apply_sobel_filter(int width, int height, pixel * pi);
#ifdef __cplusplus
}
#endif

// void apply_sobel_filter(int width, int height, pixel * pi);

