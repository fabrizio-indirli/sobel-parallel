#include "mpi_mode_1.h"

#define IN_DEBUG 0

int rank;

void apply_all_filters_mode1(int * ws, int * hs, pixel ** p, int num_subimgs, 
                        MPI_Request * reqs, MPI_Datatype mpi_pixel_type){
    int i, width, height;
    for ( i = 0 ; i < num_subimgs ; i++ )
    {
        #if IN_DEBUG
            printf("\nProcess %d is applying filters on image %d of %d\n",
            rank, i, num_subimgs);
        #endif
        pixel * pi = p[i];
        width = ws[i];
        height = hs[i];

        //MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);

        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter_omp(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter_omp( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        apply_sobel_filter_omp(width, height, pi);

        /* Send back to rank 0 */
        // MPI_Isend(pi, width*height, mpi_pixel_type, 0, 3, MPI_COMM_WORLD, &(reqs[i]));
        MPI_Send(pi, width*height, mpi_pixel_type, 0, 3, MPI_COMM_WORLD);

    }
}


void apply_all_filters0_mode1(int * ws, int * hs, pixel ** p, int num_subimgs){
    int i, width, height;
    for ( i = 0 ; i < num_subimgs ; i++ )
    {
        #if IN_DEBUG
            printf("\nProcess %d is applying filters on image %d of %d\n",
            rank, i, num_subimgs);
        #endif
        pixel * pi = p[i];
        width = ws[i];
        height = hs[i];


        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter_omp(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter_omp( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        apply_sobel_filter_omp(width, height, pi);

    }
}


void useMPIonImgs(MPI_Datatype mpi_pixel_type, int num_nodes, 
                    animated_gif * image, int my_rank)
{   
    rank = my_rank;
    struct timeval t0, t1, t2;

    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i, j;
    int n_imgs_this_node;


    int num_imgs = 0;

    if(my_rank == 0){
        // work scheduling done by first node
        pixel ** p ;
        p = image->p ;
        num_imgs = image->n_images;

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
            // if there's a rest to this division, it's added to the number of
            // images processed by the first ranks.
            #define NPN (num_imgs/num_nodes) //integer division
            int rest = num_imgs % num_nodes;
            for(i = 0; i < num_nodes; i++){
                if(rest>0) {n_imgs_per_node[i] = NPN + 1; rest--;}
                else n_imgs_per_node[i] = NPN;
            }
        }


        #if IN_DEBUG
            printf("\nThe %d ranks will process the following number of sub-imgs each: ", num_nodes);
            printVector(n_imgs_per_node, num_nodes);
        #endif
        
        int n_prev_imgs = n_imgs_per_node[0];

        #define W0 image->width[n_prev_imgs]
        #define H0 image->height[n_prev_imgs]

        // MPI_Request ** reqsPerNode = (MPI_Request **)malloc(num_nodes*sizeof(MPI_Request *));

        

        for(i=1; i<num_nodes; i++){
            
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

            //requests vector
            MPI_Request reqs[n_imgs_per_node[i]];

            //send pixels to other processes
            for(j=0; j < n_imgs_per_node[i]; j++){ 
                MPI_Isend(p[n_prev_imgs], W0*H0, mpi_pixel_type, i,2, MPI_COMM_WORLD, &reqs[j]);
                n_prev_imgs++;

                #if IN_DEBUG
                    printf("\nRank 0 has sent picture %d of %d to rank %d", j, n_imgs_per_node[i], i);
                #endif
            }
                    

                       
            
        }

        printf("\nRank 0 has sent to all the nodes\n");

        // node 0 computes filters on its images
        apply_all_filters0_mode1(image->width, image->height, p, n_imgs_per_node[0]);

        // macros to extract images' sizes now that the dims vector is not available
        
        n_prev_imgs = n_imgs_per_node[0];

        //requests vector
        MPI_Request reqs[num_imgs - n_prev_imgs];

        // receive images from all the other nodes
        for(i=1; i < num_nodes; i++){
            for(j=0; j < n_imgs_per_node[i]; j++){
                #define REC_INDEX (n_prev_imgs - n_imgs_per_node[0])
                MPI_Irecv(p[n_prev_imgs], W0*H0, mpi_pixel_type, i,3, MPI_COMM_WORLD, &reqs[REC_INDEX]);
                #if IN_DEBUG
                    printf("Rank 0 has received img %d from node %d\n", j, i);
                #endif
                n_prev_imgs++;
            }
        }
        MPI_Waitall((num_imgs - n_imgs_per_node[0]), reqs, MPI_STATUSES_IGNORE);
        


    } else {
        // NODES WITH RANK >= 1

            #define WID(j) dims[j]
            #define HEI(j) dims[n_imgs_this_node + j]


            // nodes with rank >= 1 receive the number of images they have to process
            MPI_Recv(&n_imgs_this_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #if IN_DEBUG
                printf("Rank %d has received num of imgs\n", my_rank);
            #endif

            if(n_imgs_this_node > 0){

                // nodes with rank >= 1 receive the dimensions vector and the pixels matrix
                int dims[2 * n_imgs_this_node];
                MPI_Recv(dims, 2 * n_imgs_this_node, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                #if IN_DEBUG
                    printf("Rank %d has received dims vector\n", my_rank);
                #endif
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

                //requests vector
                MPI_Request reqs[n_imgs_this_node];

                //receive images to process
                for(i=0; i<n_imgs_this_node; i++){
                    // MPI_Irecv((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 2, MPI_COMM_WORLD, &reqs[i]);
                    MPI_Recv((p_rec[i]), WID(i)*HEI(i), mpi_pixel_type, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                #if IN_DEBUG
                    printf("Rank %d has received all iamges\n", my_rank);
                #endif
                
                // other node computes filters on its images and send them back to rank 0
                apply_all_filters_mode1(dims, &(dims[n_imgs_this_node]), p_rec, n_imgs_this_node, reqs, mpi_pixel_type);


            }
                
    }

}