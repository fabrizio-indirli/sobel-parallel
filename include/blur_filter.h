#include "datastr.h"
#include <stdlib.h>

void
apply_blur_filter( int width, int height, pixel * pi, int size, int threshold );

__global__ void compute_blur_filter(pixel* newP, pixel* pi, int height, 
                                    int width, int size, int * end, int threshold, int * sync);