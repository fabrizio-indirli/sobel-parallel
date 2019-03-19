#include "blur_filter.h"


#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)


void
apply_blur_filter( int width, int height, pixel * pi, int size, int threshold ) // 5, 20
{   /*This version of the sobel filter works only on one image at a time*/

    int n_iter = 0 ;
    int end = 0 ;
    int j, k ;
    pixel * new ;

   /* Allocate array of new pixels */
    new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    /* Perform at least one blur iteration */
    do
    {
        end = 1 ;
        n_iter++ ;
        #pragma omp parallel default(none) private(j,k) shared(size,threshold,width,height,pi,new,end)
        {
            #pragma omp for collapse(2) schedule(dynamic, width) 
            for(j=size; j<height-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    if(j < height/10-size || j >= height*0.9+size) // Top and Bottom
                    {
                        int stencil_j, stencil_k ;
                        int t_r = 0 ;
                        int t_g = 0 ;
                        int t_b = 0 ;

                        for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                        {
                            for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                            {
                                t_r += pi[CONV(j+stencil_j,k+stencil_k,width)].r ;
                                t_g += pi[CONV(j+stencil_j,k+stencil_k,width)].g ;
                                t_b += pi[CONV(j+stencil_j,k+stencil_k,width)].b ;
                            }
                        }

                        new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                    }
                    else // Middle
                    {
                        new[CONV(j,k,width)].r = pi[CONV(j,k,width)].r ; 
                        new[CONV(j,k,width)].g = pi[CONV(j,k,width)].g ; 
                        new[CONV(j,k,width)].b = pi[CONV(j,k,width)].b ; 
                    }
                }
            }
   
            #pragma omp for collapse(2) schedule(dynamic,width) reduction(&&:end)
            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    float diff_r ;
                    float diff_g ;
                    float diff_b ;

                    diff_r = (new[CONV(j  ,k  ,width)].r - pi[CONV(j  ,k  ,width)].r) ;
                    diff_g = (new[CONV(j  ,k  ,width)].g - pi[CONV(j  ,k  ,width)].g) ;
                    diff_b = (new[CONV(j  ,k  ,width)].b - pi[CONV(j  ,k  ,width)].b) ;

                    if ( diff_r > threshold || -diff_r > threshold 
                            ||
                                diff_g > threshold || -diff_g > threshold
                                ||
                                diff_b > threshold || -diff_b > threshold
                        ) {
                        end = 0 ;
                    }

                    pi[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                    pi[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                    pi[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
                }
            }
        } // omp parallel
    } while ( threshold > 0 && !end ) ;

    // printf( "Nb iter for image %d\n", n_iter ) ;int i=0; i<N; i++)

    free (new) ;

}
