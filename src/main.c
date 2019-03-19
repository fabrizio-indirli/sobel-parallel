/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <gif_lib.h>
#include <stdbool.h>
#include <mpi.h>
#include <stddef.h>

#include "load_pixels.h"
#include "store_pixels.h"
#include "grey_filter.h"
#include "blur_filter.h"
#include "datastr.h"
#include "helpers.h"
#include "mpi_mode_0.h"
#include "mpi_mode_1.h"
#include "mpi_mode_2.h"
#include "mpi_mode_3.h"

#define SOBELF_DEBUG 0
#define LOGGING 1
#define EXPORT 1
#define MPI_DEBUG 1
#define GDB_DEBUG 0


#if LOGGING
    #define LOG_FILENAME "./logs_plots/plog_hybrid_n4_N1-new4.csv"
#endif

int i, j;

// MPI data
int num_nodes, my_rank;

// 0: no MPI; 1: MPI on subimgs;  2: MPI on pixels;  3: hybrid
int mpi_mode;

// minimum number of avg pixels to use MPI on pixels (parts of the image)
#define MPI_PIXELS_THRESHOLD 800000

// minimum number of avg pixels to use MPI on sub-images
#define MPI_IMGS_THRESHOLD 300000

// info on GIF file
int num_imgs = 0;
pixel ** p ;


#ifdef MPI_VERSION
    MPI_Status comm_status;
#endif




int main( int argc, char ** argv )
{

    char * input_filename ; 
    char * output_filename ;
    animated_gif * image ;
    struct timeval t0, t1, t2;
    double duration ;

    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }


    #ifdef MPI_VERSION
        /* If MPI is enabled */
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    
        #if GDB_DEBUG
            if(my_rank==0) {
                // hangs program: used only for debug purposes
                int plop=1;
                printf("my pid=%d\n", getpid());
                while(plop==0) ;
            }
        #endif
    
        /*create a MPI type for struct pixel */
        #define N_ITEMS_PIXEL  3
        int blocklengths[3] = {1,1,1};
        MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};
        MPI_Datatype mpi_pixel_type;
        MPI_Aint offsets[3];
        offsets[0] = offsetof(pixel,r);
        offsets[1] = offsetof(pixel,g);
        offsets[2] = offsetof(pixel,b);
        MPI_Type_create_struct(N_ITEMS_PIXEL, blocklengths, offsets, types, &mpi_pixel_type);
        MPI_Type_commit(&mpi_pixel_type);

    #else
        num_nodes = 1;
        // my_rank = 0;
        mpi_mode = 0;
    #endif

    #if MPI_DEBUG
        //printf("Rank %d has pid=%d\n", my_rank, getpid());
    #endif
    
    /*Open perfomance log file for debug*/
    #if LOGGING
        openLogFile(LOG_FILENAME);
    #endif

    if(my_rank == 0){
        // Only initial process loads the file

        input_filename = argv[1] ;
        output_filename = argv[2] ;
        gettimeofday(&t1, NULL); /* IMPORT Timer start */

        /* Load file and store the pixels in array */
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        gettimeofday(&t2, NULL); /* IMPORT Timer stop */

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
                input_filename, image->n_images, duration ) ;
        #if LOGGING
            newRowWithFilename(input_filename);
            appendNumToRow(image->n_images);
            appendNumToRow(image->width[0]);
            appendNumToRow(image->height[0]);
            appendNumToRow(duration);
        #endif

        p = image->p ;
        num_imgs = image->n_images;
        printf("This GIF has %d sub-images\n", num_imgs);

        // get average size of subimages
        long avg_size = avgSize(image->width, image->height, num_imgs);
        printf("Average size of images is %ld pixels\n", avg_size);

        mpi_mode = selectMPImode(num_nodes, num_imgs, avg_size, MPI_IMGS_THRESHOLD, MPI_PIXELS_THRESHOLD);
        printf("Selected MPI mode %d\n", mpi_mode);
        

    }

    

    if(num_nodes>1){
        // broadcast mpi_mode
        MPI_Bcast(&mpi_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    if(mpi_mode==0){
        // if MPI is not being used, all the ranks except 0 can stop
        MPI_Finalize();

        if(my_rank > 0) return;
    }

    
    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    

    /***** Start of parallelized version of filters *****/

    switch(mpi_mode){
        case 0: compute_without_MPI(num_nodes, image, my_rank); break; 
        case 1: useMPIonImgs(mpi_pixel_type, num_nodes, image, my_rank); break;
        case 2: useMPIonPixels(mpi_pixel_type, num_nodes, image, my_rank); break;
        default: break;
    }

    



    #ifdef MPI_VERSION
        if(num_nodes > 1 && mpi_mode > 0) MPI_Finalize();
    #endif

    if(my_rank > 0) return 0;
    /***** End of parallelized version of filters *****/

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t0.tv_sec)+((t2.tv_usec-t0.tv_usec)/1e6);
    printf( "All filters done in %lf s on %d sub-images\n", duration, num_imgs) ;
    #if LOGGING
        appendNumToRow(duration);
    #endif

    #if EXPORT

        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if ( !store_pixels( output_filename, image ) ) { return 1 ; }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "Export done in %lf s in file %s\n\n", duration, output_filename ) ;
        #if LOGGING
            appendNumToRow(duration);
        #endif

        /*Close perfomance log file*/
        #if LOGGING
            closeLogFile();
        #endif
    #endif

    return 0 ;
}
