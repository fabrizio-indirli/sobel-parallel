#include "datastr.h"
#include <stdlib.h>
#include <math.h>

void apply_sobel_filter(int width, int height, pixel * pi);

void apply_sobel_filter_omp(int width, int height, pixel * pi);

void apply_sobel_filter_part(int width, int height, pixel * pi, int startheight, int finalheight);