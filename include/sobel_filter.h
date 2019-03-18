#include "datastr.h"
#include <stdlib.h>
#include <math.h>

void apply_sobel_filter(int width, int height, pixel * pi);

__global__ void compute_sobel_filter(pixel* sobel, pixel* pi, int height, int width);