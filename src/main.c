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
#include "sobel_filter.h"
#include "datastr.h"

#define SOBELF_DEBUG 1
#define LOGGING 1

#if LOGGING
    #define FILE_NAME "./logs_plots/write_plog3.csv"
    FILE *fOut;

    void writeNumToLog(double n){
        fprintf(fOut, "%lf\n", n);
    }

    void appendNumToRow(double n){
        fprintf(fOut, "%lf,",n);
    }

    void newRow(){
        fprintf(fOut, "\n");
    }

#endif

void apply_all_filters(int * sts, pixel ** p, int num_subimgs){
    int i, width, height;
    for ( i = 0 ; i < num_subimgs ; i++ )
    {
        pixel * pi = p[i];
        width = sts[i];
        height = sts[num_subimgs + i];

        /* Apply sobel filter on pixels */
        apply_sobel_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter( width, height, pi, 5, 20 ) ;

        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter(width, height, pi);

    }
}

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

    int num_nodes, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_Rank(MPI_COMM_WORLD, &my_rank);

    /*create a MPI type for struct pixel */
    #define N_ITEMS_PIXEL  3
    int blocklengths[3] = {1,1,1};
    MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};
    MPI_Datatype mpi_pixel_type;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(pixel,r);
    offsets[1] = offsetof(pixel,g);
    offsets[0] = offsetof(pixel,b);
    MPI_Type_create_struct(N_ITEMS_PIXEL, blocklengths, offsets, types, &mpi_pixel_type);
    MPI_Type_commit(mpi_pixel_type);

    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /*Open perfomance log file for debug*/
    #if LOGGING
        fOut = fopen(FILE_NAME,"a");
        if(ftell(fOut)==0) //file is empty
            fprintf(fOut, "import_time,filters_time,export_time,");
        newRow();
    #endif

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;
    #if LOGGING
        appendNumToRow(duration);
    #endif

    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i;

    pixel ** p ;
    p = image->p ;
    int num_imgs = image->n_images;

    int n_imgs_per_node = num_imgs / num_nodes; //integer division

    if(my_rank == 0){
        // work scheduling done by first node
        int n_imgs_init_node = num_imgs - (n_imgs_per_node * (num_nodes - 1));
        
        pixel ** pts;
        int sts[2*n_imgs_per_node]; //vector 'sizes to send'
        int j;

        for(i=1; i<num_nodes; i++){
            //TODO: send pixels to other processes
            pts = p[i * n_imgs_per_node];
            MPI_Send(pts, n_imgs_per_node, mpi_pixel_type, i, 0, MPI_COMM_WORLD);

            for(j=0; j < n_imgs_per_node; j++){
                sts[j] = image->width[i*n_imgs_per_node + j];
                sts[n_imgs_per_node + j] = image->height[i*n_imgs_per_node + j];
            }
            MPI_Send(sts, 2*n_imgs_per_node, MPI_INT, i, 1, MPI_COMM_WORLD);

        }

        n_imgs_per_node = n_imgs_init_node;
    }

    int width, height ;

    /***** End of parallelized version of filters *****/

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t0.tv_sec)+((t2.tv_usec-t0.tv_usec)/1e6);
    printf( "All filters done in %lf s on %d sub-images\n", duration, num_imgs ) ;
    #if LOGGING
        appendNumToRow(duration);
    #endif

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;
    #if LOGGING
        appendNumToRow(duration);
    #endif

    /*Close perfomance log file*/
    #if LOGGING
        fclose(fOut);
    #endif

    return 0 ;
}
