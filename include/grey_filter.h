#include "datastr.h"
#include <stdlib.h>
#ifndef _OPENMP
    #include <omp.h>
#endif

void
apply_gray_filter( int width, int height, pixel * pi );