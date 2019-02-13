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
#define ENABLED 0
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

void apply_all_filters(int * ws, int * hs, pixel ** p, int num_subimgs){
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


    #ifdef MPI_VERSION
        /* If MPI is enabled */
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        printf("\nHere");
    
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
        MPI_Type_commit(&mpi_pixel_type);
    #else
        num_nodes = 1;
        my_rank = 0;
    #endif

    
    /*Open perfomance log file for debug*/
    #if LOGGING
        fOut = fopen(FILE_NAME,"a");
        if(ftell(fOut)==0) //file is empty
            fprintf(fOut, "import_time,filters_time,export_time,");
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
            appendNumToRow(duration);
        #endif
    }

    

    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i, j;
    int n_imgs_per_node;

    #define WID(j) dims[j]
    #define HEI(j) dims[(n_imgs_per_node)+(j)]
    #define N_PREV_IMGS(i) (n_imgs_init_node)+((i-1)*n_imgs_per_node)

    if(my_rank == 0){
        // work scheduling done by first node
        pixel ** p ;
        p = image->p ;
        int num_imgs = image->n_images;
        printf("\nThis GIF has %d sub-images\n", num_imgs);

        n_imgs_per_node = num_imgs / num_nodes; //integer division
        if(n_imgs_per_node == 0) n_imgs_per_node = 1;

        int n_imgs_init_node = num_imgs - (n_imgs_per_node * (num_nodes - 1));

        #if MPI_DEBUG
            printf("\nFound %d MPI ranks. 1st rank will process %d imgs, the other %d imgs\n",
            num_nodes, n_imgs_init_node, n_imgs_per_node);
        #endif
        
        pixel ** pts;
        int dims[2*n_imgs_per_node]; //vector 'sizes to send'
        printf("Size of pixels matrix is: %d\n", sizeof(pixel *));


        for(i=1; i<num_nodes; i++){
            #ifdef MPI_VERSION
                //send number of images
                MPI_Send(&n_imgs_per_node, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                // send dimensions to other processes
                for(j=0; j < n_imgs_per_node; j++){
                    dims[j] = image->width[N_PREV_IMGS(i) + j];
                    dims[n_imgs_per_node + j] = image->height[N_PREV_IMGS(i) + j];
                }
                // send a vector whose first half contains the widths and whose last half contains the heights
                MPI_Send(dims, 2*n_imgs_per_node, MPI_INT, i, 1, MPI_COMM_WORLD);

                //send pixels to other processes
                for(j=0; j < n_imgs_per_node; j++){
                    MPI_Send(p[N_PREV_IMGS(i) + j], WID(j)*HEI(j), mpi_pixel_type, i,2, MPI_COMM_WORLD);
                }
            #endif
        }

        // node 0 computes filters on its images
        apply_all_filters(image->width, image->height, p, n_imgs_init_node);

    } else {
            #ifdef MPI_VERSION
                // nodes with rank >= 1 receive the number of images they have to process
                MPI_Status comm_status;
                MPI_Recv(&n_imgs_per_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &comm_status);

                // nodes with rank >= 1 receive the dimensions vector and the pixels matrix
                int dims[2 * n_imgs_per_node];
                MPI_Recv(dims, 2 * n_imgs_per_node, MPI_INT, 0, 1, MPI_COMM_WORLD, &comm_status);
                int total_num_pixels = 0;
                for(j=0; j < n_imgs_per_node; j++){
                    total_num_pixels += dims[j] * dims[n_imgs_per_node + j]; 
                }
                // allocate array of pointers to pixels' vectors
                pixel ** p_rec = (pixel **)malloc(sizeof(pixel *) * total_num_pixels);

                //allocate array of pixels for each image
                for(j=0; j<n_imgs_per_node; j++){
                    p_rec[j] = (pixel *)malloc( dims[j] * dims[n_imgs_per_node + j] * sizeof( pixel ) ) ;
                }

                //receive images to process
                for(i=0; i<n_imgs_per_node; i++){
                    MPI_Recv(&(p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 2, MPI_COMM_WORLD, &comm_status);
                }
                
                // other node computes filters on its images
                apply_all_filters(dims, &(dims[n_imgs_per_node]), p_rec, n_imgs_per_node);

            #endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process with rank %d has passed the barrier\n", my_rank);
    /***** End of parallelized version of filters *****/

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t0.tv_sec)+((t2.tv_usec-t0.tv_usec)/1e6);
    printf( "All filters done in %lf s on %d sub-images\n", duration, n_imgs_per_node ) ;
    #if LOGGING
        appendNumToRow(duration);
    #endif

    #if ENABLED

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
