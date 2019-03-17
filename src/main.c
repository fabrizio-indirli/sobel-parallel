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

#define SOBELF_DEBUG 0
#define LOGGING 0
#define EXPORT 1
#define MPI_DEBUG 0
#define GDB_DEBUG 0

#if LOGGING
    #define FILE_NAME "./logs_plots/plog_ser.csv"
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

#ifdef MPI_VERSION
    MPI_Status comm_status;
#endif




void printVector(int * v, int n){
    int i;
    printf("[");
    for(i=0; i<n; i++){
        printf(" %d ", v[i]);
    }
    printf("]\n");
}

void printHexVector(pixel ** v, int n){
    int i;
    printf("[");
    for(i=0; i<n; i++){
        printf(" %#x ", v[i]);
    }
    printf("]\n");
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

        printf("Rank %d has pid=%d\n", my_rank, getpid());
    
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
   
    int num_imgs = 0;
    pixel ** p ;

    if(my_rank == 0){
        // work scheduling done by first node
        p = image->p ;
        num_imgs = image->n_images;
        printf("This GIF has %d sub-images\n", num_imgs);

    }

    // send num of imgs to all the nodes
    MPI_Bcast(&num_imgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #if MPI_DEBUG
        printf("Rank %d:  num_imgs broadcast\n", my_rank);
    #endif

    if(my_rank > 0)
        p = (pixel **)malloc(sizeof(pixel *) * num_imgs);
    
    // build sizes vector and send it to everyone
    int dims[2*num_imgs];
    if(my_rank == 0){
        for(j=0; j<num_imgs; j++) {
            dims[j] = image->width[j];
            dims[num_imgs + j] = image->height[j];
        }
    }
    MPI_Bcast(dims, 2*num_imgs, MPI_INT, 0, MPI_COMM_WORLD);
    #if MPI_DEBUG
        printf("Rank %d:  dims broadcast\n", my_rank);
    #endif

    // TODO: controllare inversione anche nell'altro branch
    #define WIDTH(j) dims[j]
    #define HEIGHT(j) dims[j + num_imgs]

    for(j=0; j<num_imgs && my_rank > 0; j++){
        p[j] = (pixel *)malloc(WIDTH(j) * HEIGHT(j)  * sizeof( pixel ) ) ;
    }

    
    #define H_REST(j) (HEIGHT(j) % num_nodes)

    // print img dims
    // for(j=0; j < num_imgs; j++){
    //     printf("Rank %d: image %d has width %d and height %d\n", 
    //             my_rank, j, WIDTH(j), HEIGHT(j));
    // }
    

    // line from which this node has to start computing, for each image
    int my_start_heights[num_imgs];

    // num of lines (height) that this node has to compute on each image 
    int my_part_heights[num_imgs];

    #define MY_FINAL_HEIGHT(j) (my_start_heights[j] + my_part_heights[j])

    // number of lines (height) that rank r has to compute on image j
    #define PART_HEIGHT(j,r) ((HEIGHT(j)/num_nodes) + ((r < H_REST(j)) ? 1 : 0))

    // line (height) from which rank r has to start computation on image j
    #define START_HEIGHT(j,r) ((PART_HEIGHT(j,r) * r) + ((r < H_REST(j)) ? 0 : H_REST(j)))

    // index of the pixel from which rank r has to start computation on image j
    #define START_PIXEL(j,r) (START_HEIGHT(j,r) * WIDTH(j))

    // populate vectors
    for(j=0; j < num_imgs; j++)
    {
        my_part_heights[j] = PART_HEIGHT(j, my_rank);
        my_start_heights[j] = START_HEIGHT(j, my_rank);
    }

    // printf("Rank %d, my_parts_heights: ", my_rank); printVector(my_part_heights, num_imgs);
    // printf("Rank %d, my_start_heights: ", my_rank); printVector(my_start_heights, num_imgs);


    int displs[num_imgs][num_nodes];
    int partsSizes[num_imgs][num_nodes];
    for(j=0; (j < num_imgs && my_rank==0); j++){
        for(i=0; i<num_nodes; i++){
            displs[j][i] = START_HEIGHT(j,i) * WIDTH(j);
            partsSizes[j][i] = PART_HEIGHT(j,i) * WIDTH(j);
        }
    }
   
    

    //requests vector
    MPI_Request reqs[num_imgs];
    MPI_Request reqs_final[num_imgs][num_nodes];
    for(j=0; j < num_imgs; j++){
        for(i=0; i < num_nodes; i++) reqs_final[j][i] = MPI_REQUEST_NULL;
        reqs[j] = MPI_REQUEST_NULL;
    }
    MPI_Request req;

    // Broadcast all images
    for(i=0; i < num_imgs; i++){
        MPI_Ibcast(p[i], WIDTH(i)*HEIGHT(i), mpi_pixel_type, 0, MPI_COMM_WORLD, &reqs[i]);
        // MPI_Bcast(p[i], WIDTH(i)*HEIGHT(i), mpi_pixel_type, 0, MPI_COMM_WORLD);
    }

    // apply filters and send back to rank 0
    int width, height;
    for ( j = 0 ; j < num_imgs ; j++ )
    {
        // wait for image j
        MPI_Wait(&reqs[j], MPI_STATUS_IGNORE);

        pixel * pj = p[j];
        width = dims[j];
        height = HEIGHT(j);

        #if MPI_DEBUG
            printf("\nProcess %d is applying filters on image %d of %d with width: %d  and height: %d\n",
            my_rank, j, num_imgs, width, height);
        #endif
        


        // /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter_part(width, height, pj, my_start_heights[j], MY_FINAL_HEIGHT(j));

        // /*Apply blur filter with convergence value*/
        apply_blur_filter_part( width, height, pj, 5, 20, my_start_heights[j], MY_FINAL_HEIGHT(j) ) ;

        // /* Apply sobel filter on pixels */
        apply_sobel_filter_part(width, height, pj, my_start_heights[j], MY_FINAL_HEIGHT(j));



        /* Send back to rank 0 */
        #if MPI_DEBUG
            printf("Rank %d: preparing to gather for img %d which has computed %d lines\n", my_rank, j, PART_HEIGHT(j, my_rank));
        #endif
        MPI_Igatherv(&pj[START_PIXEL(j, my_rank)], width * PART_HEIGHT(j, my_rank), mpi_pixel_type, pj, 
                    partsSizes[j], displs[j], mpi_pixel_type, 0, MPI_COMM_WORLD, reqs_final[j]);

        // if(my_rank == 0) {printf("Displacements for image %d: ", j); printVector(displs[j], num_nodes);}

        // MPI_Gatherv(&pj[START_PIXEL(j, my_rank)], width * PART_HEIGHT(j, my_rank), mpi_pixel_type, pj, 
        //             partsSizes[j], displs[j], mpi_pixel_type, 0, MPI_COMM_WORLD);
        
        // if(my_rank > 0) MPI_Send(&pj[START_PIXEL(j, my_rank)] , width * PART_HEIGHT(j, my_rank),
        //                             mpi_pixel_type, 0, 6, MPI_COMM_WORLD);
        // else {
        //     for(i=1; i < num_nodes; i++){
        //     MPI_Recv(&pj[START_PIXEL(j, my_rank)], width * PART_HEIGHT(j, i), mpi_pixel_type, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     }
        // }

        
    }



    #ifdef MPI_VERSION
        for ( j = 0 ; j < num_imgs ; j++ ) 
            MPI_Waitall(num_nodes, reqs_final[j], MPI_STATUSES_IGNORE);
        MPI_Finalize();
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
            fclose(fOut);
        #endif
    #endif

    return 0 ;
}
