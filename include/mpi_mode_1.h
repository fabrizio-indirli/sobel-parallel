#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <stdbool.h>
#include <mpi.h>
#include <stddef.h>
#include "helpers.h"


void apply_all_filters_mode1(int * ws, int * hs, pixel ** p, int num_subimgs, 
                        MPI_Request * reqs, MPI_Datatype mpi_pixel_type);

void apply_all_filters0_mode1(int * ws, int * hs, pixel ** p, int num_subimgs);

void useMPIonImgs(MPI_Datatype mpi_pixel_type, int num_nodes,  
                    animated_gif * image, int my_rank);

