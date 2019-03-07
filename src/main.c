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
#define LOGGING 1
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


        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        apply_sobel_filter(width, height, pi);

    }
}

void printVector(int * v, int n){
    int i;
    for(i=0; i<n; i++){
        printf(" %d ", v[i]);
    }
    printf("\n");
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
    int n_imgs_this_node;

    #define WID(j) dims[j]
    #define HEI(j) dims[n_imgs_this_node + j]

    #ifdef MPI_VERSION
        MPI_Status comm_status;
    #endif

    int num_imgs = 0;

    if(my_rank == 0){
        // work scheduling done by first node
        pixel ** p ;
        p = image->p ;
        num_imgs = image->n_images;
        printf("\nThis GIF has %d sub-images\n", num_imgs);

        int n_imgs_per_node[num_nodes];

        int i;
        // compute num of imgs that each node has to process
        if(num_nodes > num_imgs){
            // if there are more ranks than images, each rank processes 1 image
            printf("Too many nodes: %d of them won't be used", (num_nodes - num_imgs));
            for(i = 0; i < num_nodes; i++){
                if(i < num_imgs) n_imgs_per_node[i] = 1;
                else n_imgs_per_node[i] = 0;
            }
        } else {
            // otherwise, each rank processes (num_imgs / num_nodes) images.
            // if ther's a rest to this division, it's added to the number of
            // images processed by the first ranks.
            #define NPN (num_imgs/num_nodes) //integer division
            int rest = num_imgs % num_nodes;
            for(i = 0; i < num_nodes; i++){
                if(rest>0) {n_imgs_per_node[i] = NPN + 1; rest--;}
                else n_imgs_per_node[i] = NPN;
            }
        }


        #if MPI_DEBUG
            printf("\nThe %d ranks will process the following number of sub-imgs each: ", num_nodes);
            printVector(n_imgs_per_node, num_nodes);
        #endif
        
        int n_prev_imgs = n_imgs_per_node[0];

        #define W0 image->width[n_prev_imgs]
        #define H0 image->height[n_prev_imgs]

        for(i=1; i<num_nodes; i++){
            #ifdef MPI_VERSION
                
                    int dims[2*n_imgs_per_node[i]]; //vector 'sizes to send'
                    
                    //send number of images
                    MPI_Send(&(n_imgs_per_node[i]), 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                    // send dimensions to other processes
                    for(j=0; j < n_imgs_per_node[i]; j++){
                        dims[j] = image->width[n_prev_imgs + j];
                        dims[n_imgs_per_node[i] + j] = image->height[n_prev_imgs + j];
                    }
                    // send a vector whose first half contains the widths and whose last half contains the heights
                    MPI_Send(dims, 2*n_imgs_per_node[i], MPI_INT, i, 1, MPI_COMM_WORLD);

                    //send pixels to other processes
                    for(j=0; j < n_imgs_per_node[i]; j++){ 
                        MPI_Send(p[n_prev_imgs], W0*H0, mpi_pixel_type, i,2, MPI_COMM_WORLD);
                        n_prev_imgs++;
                    }

                       
            #endif
        }

        // node 0 computes filters on its images
        apply_all_filters(image->width, image->height, p, n_imgs_per_node[0]);

        #ifdef MPI_VERSION
            // macros to extract images' sizes now that the dims vector is not available
            
            n_prev_imgs = n_imgs_per_node[0];


            // receive images from all the other nodes
            for(i=1; i < num_nodes; i++){
                for(j=0; j < n_imgs_per_node[i]; j++){
                    MPI_Recv(p[n_prev_imgs], W0*H0, mpi_pixel_type, i,3, MPI_COMM_WORLD, &comm_status);
                    n_prev_imgs++;
                }
            }
        #endif


    } else {
        // NODES WITH RANK >= 1

            #ifdef MPI_VERSION

                // nodes with rank >= 1 receive the number of images they have to process
                MPI_Recv(&n_imgs_this_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &comm_status);

                if(n_imgs_this_node > 0){

                    // nodes with rank >= 1 receive the dimensions vector and the pixels matrix
                    int dims[2 * n_imgs_this_node];
                    MPI_Recv(dims, 2 * n_imgs_this_node, MPI_INT, 0, 1, MPI_COMM_WORLD, &comm_status);
                    int total_num_pixels = 0;
                    for(j=0; j < n_imgs_this_node; j++){
                        total_num_pixels += dims[j] * dims[n_imgs_this_node + j]; 
                    }
                    // allocate array of pointers to pixels' vectors
                    pixel ** p_rec = (pixel **)malloc(sizeof(pixel *) * total_num_pixels);

                    //allocate array of pixels for each image
                    for(j=0; j<n_imgs_this_node; j++){
                        p_rec[j] = (pixel *)malloc( dims[j] * dims[n_imgs_this_node + j] * sizeof( pixel ) ) ;
                    }

                    //receive images to process
                    for(i=0; i<n_imgs_this_node; i++){
                        MPI_Recv((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 2, MPI_COMM_WORLD, &comm_status);
                    }
                    
                    // other node computes filters on its images
                    apply_all_filters(dims, &(dims[n_imgs_this_node]), p_rec, n_imgs_this_node);

                    //send back to node 0 the processed images
                    for(i=0; i<n_imgs_this_node; i++){
                        MPI_Send((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 3, MPI_COMM_WORLD);
                    }

                }
                

            #endif
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
