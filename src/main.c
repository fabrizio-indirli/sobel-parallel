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
                        int start_img, MPI_Datatype mpi_pixel_type){
    int i, width, height;
    MPI_Request req;

    for ( i = start_img ; i < start_img + num_subimgs ; i++ )
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
        if(my_rank != 0)
            MPI_Isend(pi, width*height, mpi_pixel_type, 0, 3, MPI_COMM_WORLD, &req);

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


    // Only initial process loads the file

    input_filename = argv[1] ;
    output_filename = argv[2] ;
    gettimeofday(&t1, NULL); /* IMPORT Timer start */

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    gettimeofday(&t2, NULL); /* IMPORT Timer stop */

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "Rank %d has loaded from file %s with %d image(s) in %lf s\n", 
            my_rank, input_filename, image->n_images, duration ) ;
    #if LOGGING
        appendNumToRow(image->n_images);
        appendNumToRow(image->width[0]);
        appendNumToRow(image->height[0]);
        appendNumToRow(duration);
    #endif
    
    #if GDB_DEBUG
        if(my_rank==1) {
            // hangs program: used only for debug purposes
            int plop=0;
            printf(" pid of rank %d is %d\n", my_rank, getpid());
            while(plop==0) ;
        }
    #endif
    
    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i, j;
    int n_imgs_this_node;

    #define WID(j) image->width[j]
    #define HEI(j) image->height[j]


    int num_imgs = image->n_images;
    pixel ** p ;
    p = image->p ;

    #define REST (num_imgs % num_nodes)
    #define PART (num_imgs / num_nodes)
    #define N_IMGS_NODE(i) ((num_nodes > num_imgs) ? ((i < num_imgs) ? 1 : 0) : ((i < REST) ? (PART + 1) : (PART)))

    n_imgs_this_node = N_IMGS_NODE(my_rank);
    int start_img = (my_rank < REST) ? (my_rank * (PART + 1)) : (my_rank * PART + REST);

    printf("\nRank %d is HERE", my_rank);
    //printf("\nRank %d has to compute picture from %d to %d", my_rank, start_img, start_img + n_imgs_this_node);
    

    //apply filters on my pictures
    apply_all_filters(image->width, image->height, p, n_imgs_this_node, start_img, mpi_pixel_type);

    if(my_rank == 0){
        // work scheduling done by first node
        printf("\nThis GIF has %d sub-images\n", num_imgs);


        #ifdef MPI_VERSION
            // macros to extract images' sizes now that the dims vector is not available
            
            int n_prev_imgs = n_imgs_this_node;

            //requests vector
            MPI_Request reqs[num_imgs - n_prev_imgs];

            // receive images from all the other nodes
            for(i=1; i < num_nodes; i++){
                for(j=0; j < N_IMGS_NODE(i); j++){
                    #define REC_INDEX (n_prev_imgs - n_imgs_this_node)
                    MPI_Irecv(p[n_prev_imgs], WID(n_prev_imgs)*HEI(n_prev_imgs), mpi_pixel_type, i,3, MPI_COMM_WORLD, &reqs[REC_INDEX]);
                    n_prev_imgs++;
                }
            }
            MPI_Waitall((num_imgs - n_imgs_this_node), reqs, MPI_STATUSES_IGNORE);
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
