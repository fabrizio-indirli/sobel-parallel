#include "datastr.h"
#include <stdlib.h>
#include <math.h>
#ifndef _OPENMP
    #include <omp.h>
#endif

void apply_sobel_filter(int width, int height, pixel * pi);