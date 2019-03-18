#include "datastr.h"
#include <stdlib.h>

void
apply_gray_filter_omp( int width, int height, pixel * pi );

void
apply_gray_filter_part( int width, int height, pixel * pi, int startheight, int finalheight );

void
apply_gray_filter( int width, int height, pixel * pi );