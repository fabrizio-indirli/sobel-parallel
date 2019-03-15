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
#define MPI_DEBUG 1
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

void apply_all_filters(int * ws, int * hs, pixel ** p, int num_subimgs, 
                        MPI_Datatype mpi_pixel_type, MPI_Request * reqs){
    int i, width, height;
    for ( i = 0 ; i < num_subimgs ; i++ )
    {
        #if MPI_DEBUG
            printf("\nProcess %d is applying filters on image %d of %d\n",
            my_rank, i, num_subimgs);
        #endif
        pixel * pi = p[i];
        width = ws[i];
        height = hs[i];


        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        apply_sobel_filter(width, height, pi);

        /* Send back to rank 0 */
        if(my_rank > 0)
            MPI_Isend(pi, width*height, mpi_pixel_type, 0, 3, MPI_COMM_WORLD, &(reqs[i]));

    }
}



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

    printf("Rank %d has pid=%d\n", my_rank, getpid());

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
    printf("Rank %d:  num_imgs broadcast\n", my_rank);
    
    // build sizes vector and send it to everyone
    int dims[2*num_imgs];
    if(my_rank == 0){
        for(j=0; j<num_imgs; j++) {
            dims[j] = image->width[j];
            dims[num_imgs + j] = image->height[j];
        }
    }
    MPI_Bcast(dims, 2*num_imgs, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d:  dims broadcast\n", my_rank);


    //compute pixels (heights) of each process
    #define HEIGHT(j) dims[j]
    #define WIDTH(j) dims[j + num_imgs]

    // print img dims
    for(j=0; j < num_imgs; j++){
        printf("Rank %d: image %d has width %d and height %d\n", 
                my_rank, j, WIDTH(j), HEIGHT(j));
    }
    
    // vectors to send (NOT CONSIDERING THE OVERLAPS)
    int hxn[num_nodes][num_imgs]; // heights per node
    int pxn[num_imgs][num_nodes]; // pixels per node
    int start_h[num_imgs][num_nodes]; // start height
    int displs[num_imgs][num_nodes]; // displacements

    if(my_rank == 0){
        int rest, height_x_node;
        

        for(j=0; j < num_imgs; j++){
            start_h[j][0] = 0;
            displs[j][0] = 0;
            rest = HEIGHT(j) % num_nodes;
            printf("Subimg %d has %d pixels, with an height of %d\n", j, WIDTH(j)*HEIGHT(j), HEIGHT(j));

            for(i=0; i< num_nodes; i++){
                hxn[i][j] = (i < rest) ? (HEIGHT(j)/num_nodes + 1) : (HEIGHT(j)/num_nodes);
                pxn[j][i] = hxn[i][j]*WIDTH(j);

                if(j < (num_imgs - 1)) {
                    start_h[j][i+1] = start_h[j][i] + hxn[i][j];
                    displs[j][i+1] = displs[j][i] + hxn[i][j]*WIDTH(j);
                }
            }
        }

        printf("Rank 0 has computed parts of images\n");
    }
    
    int pxn_this_node_sc[num_imgs]; // pixels received with scatterv for each image
    int hxn_this_node_sc[num_imgs]; // num of lines received with scatterv for each image

    // populate vector "hxn_this_node"
    MPI_Scatter(hxn, num_imgs, MPI_INT, hxn_this_node_sc, num_imgs, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d:  scatter hxn\n", my_rank);

    // populate vector "pxn_this_node"
    for(j=0; j < num_imgs; j++)
        pxn_this_node_sc[j] = WIDTH(j) * hxn_this_node_sc[j];


    pixel ** p_rec = (pixel **)malloc(sizeof(pixel *) * num_imgs);
    pixel ** p_rec_scatt_start = (pixel **)malloc(sizeof(pixel *) * num_imgs);

    
    #define HXN(i, j) (i < (num_nodes-1)) ? (hxn[i][j] + 1) : (hxn[i][j])

    // additional lines to send
    #define FIRST_LINE(j) (my_rank > 0) ? (start_h[i][j] - 1) : (start_h[i][j])
    #define LAST_LINE(j) (my_rank == (num_nodes - 1)) ? (start_h[i][j] + hxn[i][j]) : (start_h[i][j] + hxn[i][j] + 1)
    #define FL_TAG 3
    #define LL_TAG 4

    // total number of pixels to receive and store for each image, including the additional lines up and down
    // but rank 0 and last rank only take 1 additional line (respectively above and below)
    #define PIX_STORED(j) ((my_rank == 0 || my_rank == (num_nodes - 1)) ? (pxn_this_node_sc[j] + WIDTH(j)) : (pxn_this_node_sc[j] + 2*WIDTH(j)))

    //allocate array of pixels for each image
    for(j=0; j<num_imgs; j++){
        p_rec[j] = (pixel *)malloc( PIX_STORED(j) * sizeof( pixel ) ) ;
        p_rec_scatt_start[j] = (my_rank > 0) ? (&(p_rec[j][WIDTH(j)])) : p_rec[j];
    }

    int k;
    pixel * line1ToSend, * line2ToSend;

    // iterate on the images: scatter them
    for(j=0; j<num_imgs; j++){
        printf("Rank %d: preparing to scatter image %d\n", my_rank, j);

        if(my_rank==0){
            printf("Sizes for img %d: ", j); printVector(pxn[j], num_nodes);
            printf("DIsplacements for img %d: ", j); printVector(displs[j], num_nodes);
        }

        printf("p_rec vector for node %d: ", my_rank); printHexVector(p_rec, num_imgs);
        printf("p_rec_scatt_start vector for node %d: ", my_rank); printHexVector(p_rec, num_imgs);

        // send the separate parts to compute with scatterv
        MPI_Scatterv(p[j], pxn[j], displs[j], mpi_pixel_type, 
        p_rec_scatt_start[j], pxn_this_node_sc[j], mpi_pixel_type, 0, MPI_COMM_WORLD);
        
        if(my_rank==0){
            // rank 0 sends additional lines to the other nodes
            printf("\n Scattered image %d\n", j);
            line1ToSend = (pixel *)malloc(sizeof(pixel) * WIDTH(j));
            line2ToSend = (pixel *)malloc(sizeof(pixel) * WIDTH(j));

            for(i=1; i<num_nodes; i++){

                for(k=0; k<WIDTH(j); k++){
                    line1ToSend[k] = p[j][WIDTH(j) * FIRST_LINE(j) + k];
                    if(i < (num_nodes - 1)) line2ToSend[k] = p[j][WIDTH(j) * LAST_LINE(j) + k];
                    }

                // send first additional line to other ranks
                MPI_Send(line1ToSend, WIDTH(j), mpi_pixel_type, i, FL_TAG, MPI_COMM_WORLD);

                // send last additional line to other ranks (except the last one)
                if(i < (num_nodes - 1))
                    MPI_Send(line2ToSend, WIDTH(j), mpi_pixel_type, i, LL_TAG, MPI_COMM_WORLD);
     
            }
        } else {
            // ranks >= 1 receive all the additional lines

            // receive first additional line from rank 0
            MPI_Recv(p_rec[j], WIDTH(j), mpi_pixel_type, 0, FL_TAG, MPI_COMM_WORLD, &comm_status);

            // receive the last additional line, except for the last node
            if(my_rank < (num_nodes - 1))
                MPI_Recv(&p_rec[j][pxn_this_node_sc[j]], WIDTH(j), mpi_pixel_type, 0, LL_TAG, MPI_COMM_WORLD, &comm_status);

        }
    }

    if(my_rank > 0) p = p_rec;

    //requests vector
    MPI_Request reqs[num_imgs];

    // apply filters and send back to rank 0
    // apply_all_filters(dims, hxn_this_node_sc, p, num_imgs, mpi_pixel_type, reqs);
    // MPI_Waitall(num_imgs, reqs, MPI_STATUSES_IGNORE);

    int width, height;
    for ( i = 0 ; i < num_imgs ; i++ )
    {
        #if MPI_DEBUG
            printf("\nProcess %d is applying filters on image %d of %d\n",
            my_rank, i, num_imgs);
        #endif
        pixel * pi = p[i];
        width = dims[i];
        height = hxn_this_node_sc[i];


        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        apply_sobel_filter(width, height, pi);

        /* Send back to rank 0 */
        MPI_Igatherv(pi, width * height, mpi_pixel_type, pi, pxn[i], displs[i], 
                     mpi_pixel_type, 0, MPI_COMM_WORLD, reqs);

    }



    #ifdef MPI_VERSION
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
