#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <stdbool.h>
#include <mpi.h>
#include <stddef.h>

#include "helpers.h"
#include "grey_filter.h"
#include "blur_filter.h"
#include "sobel_filter.h"

#ifdef __cplusplus
extern "C"{
#endif 

void useMPIonPixels(MPI_Datatype mpi_pixel_type, int num_nodes,  
                    animated_gif * image, int my_rank);

#ifdef __cplusplus
}
#endif