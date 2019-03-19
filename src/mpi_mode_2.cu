#include "mpi_mode_2.h"

void useMPIonPixels(MPI_Datatype mpi_pixel_type, int num_nodes,  
                    animated_gif * image, int my_rank)
{
    struct timeval t0, t1, t2;

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
    #endif
}