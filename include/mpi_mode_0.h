#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <stdbool.h>
#include <omp.h>
#include <stddef.h>

#include "helpers.h"
#include "grey_filter.h"
#include "blur_filter.h"
#include "sobel_filter.h"

void compute_without_MPI(int num_nodes, animated_gif * image, int my_rank);
