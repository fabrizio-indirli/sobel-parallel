#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <stdbool.h>
#include <mpi.h>
#include <stddef.h>

#include "helpers.h"
#include "blur_filter.h"
#include "grey_filter.h"
// #include "sobel_filter.h"

extern void apply_sobel_filter(int width, int height, pixel * pi);

extern void apply_sobel_filter_omp(int width, int height, pixel * pi);

extern void apply_sobel_filter_part(int width, int height, pixel * pi, int startheight, int finalheight);

extern void sobel_filter_cuda(int width, int height, pixel * pi);

extern void sobel_filter_auto(int width, int height, pixel * pi);

extern void sobel_filter_part_auto(int width, int height, pixel * pi, int startheight, int finalheight);



#ifdef __cplusplus
extern "C"{
#endif 

void apply_all_filters_mode1(int * ws, int * hs, pixel ** p, int num_subimgs, 
                        MPI_Request * reqs, MPI_Datatype mpi_pixel_type);

void apply_all_filters0_mode1(int * ws, int * hs, pixel ** p, int num_subimgs);

void useMPIonImgs(MPI_Datatype mpi_pixel_type, int num_nodes,  
                    animated_gif * image, int my_rank);

#ifdef __cplusplus
}
#endif

