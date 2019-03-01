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

#include "load_pixels.h"
#include "store_pixels.h"
#include "grey_filter.h"
#include "blur_filter.h"
#include "sobel_filter.h"

#define SOBELF_DEBUG 1
#define LOGGING 1
#define EXPORT 1
#define MPI_DEBUG 1

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

int num_nodes, my_rank;

int cumulativeSum(int * vect, int num){
    int i;
    int sum = 0;
    for(i=0; i<num; i++){
        sum += vect[i];
    }
    return sum;
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
        my_rank = 0;
    #endif


    /*Open perfomance log file for debug*/
    #if LOGGING
        fOut = fopen(FILE_NAME,"a");
        if(ftell(fOut)==0) //file is empty
            fprintf(fOut, "n_subimgs,width,height,import_time,filters_time,export_time,");
        newRow();
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
            appendNumToRow(image->n_images);
            appendNumToRow(image->width[0]);
            appendNumToRow(image->height[0]);
            appendNumToRow(duration);
        #endif
    }

    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i, j;
    int width, height ;

    #ifdef MPI_VERSION
        MPI_Status comm_status;
    #endif

    #define N_NODES_x_FILTER (num_nodes/3)

   int num_imgs = 0;

    if(my_rank == 0){
        // work scheduling done by first node
        pixel ** p ;
        p = image->p ;
        num_imgs = image->n_images;

        // compute num of imgs to send to each node
        int n_imgs_x_node[N_NODES_x_FILTER];
        int rest = num_imgs/N_NODES_x_FILTER;

        for(j=0; j < N_NODES_x_FILTER; j++){
            
            n_imgs_x_node[j] =(rest > 0) ? (num_imgs/N_NODES_x_FILTER +1) : (num_imgs/N_NODES_x_FILTER);
            rest--;
        }

        if(num_nodes > 2){
            #define N_IMGS_NODE(i) (n_imgs_x_node[i % 3])

            int num_imgs_next_node;

            //rank 0 sents to other nodes the number of images they have to compute
            for(i=1; i<((num_nodes/3) * 3); i++){
                num_imgs_next_node = N_IMGS_NODE(i);
                int dims[2*num_imgs_next_node]; //vector 'sizes to send'

                //send number of images
                MPI_Send(&num_imgs_next_node, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                // send dimensions to other processes
                for(j=0; j < num_imgs_next_node; j++){
                    dims[j] = image->width[cumulativeSum(n_imgs_x_node, i) + j];
                    dims[num_imgs_next_node + j] = image->height[cumulativeSum(n_imgs_x_node, i) + j];
                }

                // send a vector whose first half contains the widths and whose last half contains the heights
                MPI_Send(dims, 2*num_imgs_next_node, MPI_INT, i, 1, MPI_COMM_WORLD);



            }

           // rank 0 sents images to other nodes that have to apply the first filter (grey filter)
            for(i=0; i<num_imgs; i++){

            }
        }
    }
    /***** End of parallelized version of filters *****/

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t0.tv_sec)+((t2.tv_usec-t0.tv_usec)/1e6);
    printf( "All filters done in %lf s\n", duration ) ;
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
