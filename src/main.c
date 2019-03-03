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

#define N_NODES_x_FILTER (num_nodes/3)

int cumulativeSum(int * vect, int num){
    int i;
    int sum = 0;
    for(i=0; i<num; i++){
        sum += vect[i];
    }
    return sum;
}

#define WID(i) dims[i]
#define HEI(i) dims[num_imgs_this_node + i]

void send2next(int num_imgs_this_node, pixel ** p, int * dims, MPI_Datatype mpi_pixel_type){
    printf("\nI AM HERE %d", num_nodes);

    int dest = my_rank + N_NODES_x_FILTER;

    //send number of images
    MPI_Send(&num_imgs_this_node, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

    // send a vector whose first half contains the widths and whose last half contains the heights
    MPI_Send(dims, 2*num_imgs_this_node, MPI_INT, dest, 1, MPI_COMM_WORLD);

    // send pictures to next node in the pipeline
    int j;
    for(j=0; j<num_imgs_this_node; j++){
        MPI_Send(p[j], WID(j)*HEI(j), mpi_pixel_type, dest, 2, MPI_COMM_WORLD);
    }
    printf("Rank %d has sent %d pictures to rank %d\n", my_rank, num_imgs_this_node, dest);


}

void apply_filter(int * ws, int * hs, pixel ** p, int num_subimgs){
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

        /*1st group of nodes apply grey filter: convert the pixels into grayscale */
        if(my_rank < N_NODES_x_FILTER) apply_gray_filter(width, height, pi);

        /*2nd group of nodes apply blur filter with convergence value*/
        if(my_rank >= N_NODES_x_FILTER && my_rank < 2*N_NODES_x_FILTER)
            apply_blur_filter( width, height, pi, 5, 20 ) ;
        
        /* 3rd group apply sobel filter on pixels */
        if(my_rank >= 2*N_NODES_x_FILTER) apply_sobel_filter(width, height, pi);
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


    if(my_rank == 0){
        // work scheduling done by first node
        pixel ** p ;
        p = image->p ;
        int num_imgs = image->n_images;

        // compute num of imgs to send to each node
        int n_imgs_x_node[N_NODES_x_FILTER];
        int rest = num_imgs/N_NODES_x_FILTER;

        for(j=0; j < N_NODES_x_FILTER; j++){
            
            n_imgs_x_node[j] =(rest > 0) ? (num_imgs/N_NODES_x_FILTER +1) : (num_imgs/N_NODES_x_FILTER);
            rest--;
        }

        if(num_nodes > 2){
            #define N_IMGS_NODE(i) (n_imgs_x_node[i % 3])

            int num_imgs_this_node;

            //rank 0 sents to other nodes the number of images they have to compute
            for(i=1; i<N_NODES_x_FILTER; i++){

                num_imgs_this_node = N_IMGS_NODE(i);
                int dims[2*num_imgs_this_node]; //vector 'sizes to send'

                printf("\nI AM HERE 1");

                //send number of images
                MPI_Send(&num_imgs_this_node, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                // buld dimensions vector
                for(j=0; j < num_imgs_this_node; j++){
                    dims[j] = image->width[cumulativeSum(n_imgs_x_node, i) + j];
                    dims[num_imgs_this_node + j] = image->height[cumulativeSum(n_imgs_x_node, i) + j];
                }

                // send a vector whose first half contains the widths and whose last half contains the heights
                MPI_Send(dims, 2*num_imgs_this_node, MPI_INT, i, 1, MPI_COMM_WORLD);

                #define N_PREV_IMGS(i) cumulativeSum(n_imgs_x_node, i)

                // rank 0 sents images to other nodes that have to apply the first filter (grey filter)
                for(i=1; i<N_NODES_x_FILTER; i++){
                    for(j=0; j<n_imgs_x_node[i]; j++){
                        MPI_Send(p[N_PREV_IMGS(i) + j], WID(j)*HEI(j), mpi_pixel_type, i,2, MPI_COMM_WORLD);
                    }
                    printf("\nRank 0 has sent the images to rank %d\n", i);
                }

            }

            // node 0 applies its filter on its pictures and sends to next node
            int dims[2*N_IMGS_NODE(0)]; //vector 'sizes to send'
            // buld dimensions vector
            for(j=0; j < N_IMGS_NODE(0); j++){
                dims[j] = image->width[j];
                dims[num_imgs_this_node + j] = image->height[j]; }
            apply_filter(image->width, image->height, p, N_IMGS_NODE(0));
            printf("\nI AM HERE 2");
            // send2next(N_IMGS_NODE(0), p, dims, mpi_pixel_type);

            // macros to extract images' sizes now that the dims vector is not available
            #define W0(i,j) image->width[(N_PREV_IMGS(i))+(j)]
            #define H0(i,j) image->height[(N_PREV_IMGS(i))+(j)]

            // node 0 receives from the group 3 nodes
            for(i = 2*N_NODES_x_FILTER; i<3*N_NODES_x_FILTER; i++){
                for(j=0; j < N_IMGS_NODE(i); j++){
                    MPI_Recv(p[N_PREV_IMGS(i) + j], W0(i,j)*H0(i,j), mpi_pixel_type, i,3, MPI_COMM_WORLD, &comm_status);
                }

            }

        }
    } else {
        // nodes with rank >= 1
        int num_imgs_this_node;

        // nodes with rank >= 1 receive the number of images they have to process
        MPI_Recv(&num_imgs_this_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &comm_status);

        // nodes with rank >= 1 receive the dimensions vector and the pixels matrix
        int dims[2 * num_imgs_this_node];
        MPI_Recv(dims, 2 * num_imgs_this_node, MPI_INT, 0, 1, MPI_COMM_WORLD, &comm_status);
        int total_num_pixels = 0;
        for(j=0; j < num_imgs_this_node; j++){
            total_num_pixels += dims[j] * dims[num_imgs_this_node + j]; 
        }
        // allocate array of pointers to pixels' vectors
        pixel ** p_rec = (pixel **)malloc(sizeof(pixel *) * total_num_pixels);

        //allocate array of pixels for each image
        for(j=0; j<num_imgs_this_node; j++){
            p_rec[j] = (pixel *)malloc( dims[j] * dims[num_imgs_this_node + j] * sizeof( pixel ) ) ;
        }

        //receive images to process
        for(i=0; i<num_imgs_this_node; i++){
            MPI_Recv((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 2, MPI_COMM_WORLD, &comm_status);
        }

        // PROCESS IMAGES //
        apply_filter(dims, &(dims[num_imgs_this_node]), p_rec, num_imgs_this_node);
        
        if(my_rank < 2*N_NODES_x_FILTER){
            // if this node is in the 1st or 2nd group, send to next node in the pipeline
            send2next(num_imgs_this_node, p_rec, dims, mpi_pixel_type);
        }
        else {
            // if this node is in the 3rd group, send back to node 0
            for(i=0; i<num_imgs_this_node; i++){
                MPI_Send((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 3, MPI_COMM_WORLD);
            }
        }

    }
    /***** End of parallelized version of filters *****/

    MPI_Finalize();
    if(my_rank > 0) return 0;

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
