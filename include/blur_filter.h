#include "datastr.h"
#include <stdlib.h>

void
apply_blur_filter( int width, int height, pixel * pi, int size, int threshold );


void
apply_blur_filter_omp( int width, int height, pixel * pi, int size, int threshold );


void
apply_blur_filter_part( int width, int height, pixel * pi, int size, int threshold, int partStartHeight, int partFinalHeight );