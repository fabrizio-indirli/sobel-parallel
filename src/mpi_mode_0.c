#include "mpi_mode_0.h"

void compute_without_MPI(int num_nodes, animated_gif * image, int my_rank)
{
    int i;
    int width, height ;
    double duration ;


    int num_threads=0;

    #ifdef _OPENMP
        #pragma omp parallel default(none) shared(num_threads)
        {
            #pragma omp master
            {
                num_threads = omp_get_num_threads();
                
                printf("The number of threads : %d\n", num_threads);
            }
        }
    #endif


    pixel ** p ;

    p = image->p ;

    
    
    if (image->n_images > num_threads)
    {
        // parallelized on num of images
        #pragma omp parallel default(none) private(i,width,height) shared(p,image)
        {

            #pragma omp for schedule(static,1)
            for(i = 0 ; i < image->n_images; i++)
            {   
                //printf("[FILTERS] p[%d] from thread #%d\n", i, rank);
                width = image->width[i] ;
                height = image->height[i] ;
                pixel * pi = p[i];
                int rank = 0;

                #ifdef _OPENMP
                    rank = omp_get_thread_num();
                #endif

                /*Apply grey filter: convert the pixels into grayscale */
                apply_gray_filter(width, height, pi);

                /*Apply blur filter with convergence value*/
                apply_blur_filter( width, height, pi, 5, 20 ) ;

                /* Apply sobel filter on pixels */
                // if(rank < 2)  sobel_filter_cuda(width, height, pi);
                // else apply_sobel_filter(width, height, pi);
                apply_sobel_filter(width, height, pi);
            }
            
        }
    }
    else 
    {
        for ( i = 0 ; i < image->n_images ; i++ )
        {
            // parallelization on pixels
            width = image->width[i] ;
            height = image->height[i] ;
            pixel * pi = p[i];


            /*Apply grey filter: convert the pixels into grayscale */
            apply_gray_filter_omp(width, height, pi);

            /*Apply blur filter with convergence value*/
            apply_blur_filter_omp( width, height, pi, 5, 20 ) ;

            /* Apply sobel filter on pixels */
            sobel_filter_auto(width, height, pi);
        }
    }

}