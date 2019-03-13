#include "datastr.h"
#include <stdlib.h>

void
apply_gray_filter( int width, int height, pixel * pi );

__global__ void compute_gray_filter( pixel* pi, int N );