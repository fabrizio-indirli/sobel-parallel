#include "mpi_mode_2.h"

#define MIN_PIXELS_THRESHOLD 10000

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

    int avg_size = avgSize(dims, &dims[num_imgs], num_imgs);
    
    #define MAX_RANK (avg_size/MIN_PIXELS_THRESHOLD)

    MPI_Group new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &new_group);
    MPI_Comm newworld;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &newworld);

    if(num_nodes > MAX_RANK){
        // too many ranks
        if(my_rank == 0)
            printf("Too many nodes: %d of them won't be used\n", (num_nodes - MAX_RANK));

        // Obtain the group of processes in the world communicator
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        // exclude ranks in excess
        int ranges[1][3];
        ranges[0][0] = MAX_RANK;
        ranges[0][1] = num_nodes-1;
        ranges[0][2] = 1;
        MPI_Group_range_excl(world_group, 1, ranges, &new_group);

        // Create a new communicator with less ranks
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &newworld);

        if (newworld == MPI_COMM_NULL)
        {
            // ranks in excess terminate
            MPI_Finalize();
            exit(0);
        }

        // update number of ranks
        MPI_Comm_size(newworld, &num_nodes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    }

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
        MPI_Ibcast(p[i], WIDTH(i)*HEIGHT(i), mpi_pixel_type, 0, newworld, &reqs[i]);
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
        sobel_filter_part_auto(width, height, pj, my_start_heights[j], MY_FINAL_HEIGHT(j));



        /* Send back to rank 0 */
        #if MPI_DEBUG
            printf("Rank %d: preparing to gather for img %d which has computed %d lines\n", my_rank, j, PART_HEIGHT(j, my_rank));
        #endif
        MPI_Igatherv(&pj[START_PIXEL(j, my_rank)], width * PART_HEIGHT(j, my_rank), mpi_pixel_type, pj, 
                    partsSizes[j], displs[j], mpi_pixel_type, 0, newworld, reqs_final[j]);

        
    }



    #ifdef MPI_VERSION
        for ( j = 0 ; j < num_imgs ; j++ ) 
            MPI_Waitall(num_nodes, reqs_final[j], MPI_STATUSES_IGNORE);
    #endif
}