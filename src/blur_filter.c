#include "blur_filter.h"
#include <omp.h>

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

        /* Apply blur on top part of image (10%) */
        //*** one thing that if checks only j not k...
        //*** WHAT IF not using `collapse`?
        // #pragma omp for collapse(2) schedule(static,width) 
        for(j=size; j<height-size;j++)
        {
            for(k=size; k<width-size; k++)
            {
                // int pixel = CONV(j,k,width); //*** can be useful...
                // Top and Bottom, 10% each
                if(j<height/10-size || j>height*0.9+size-1) // equivalent to (j>=height*0.9+size)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0 ;
                    int t_g = 0 ;
                    int t_b = 0 ;

                    //*** Apply Blur
                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += pi[CONV(j+stencil_j,k+stencil_k,width)].r ;
                            t_g += pi[CONV(j+stencil_j,k+stencil_k,width)].g ;
                            t_b += pi[CONV(j+stencil_j,k+stencil_k,width)].b ;
                        }
                    }

                    new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ; // (size+1) * (size+1)
                    new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;

                    //*** Now, check the threshold. 
                    float diff_r ;
                    float diff_g ;
                    float diff_b ;

                    diff_r = (new[CONV(j  ,k  ,width)].r - pi[CONV(j  ,k  ,width)].r) ;
                    diff_g = (new[CONV(j  ,k  ,width)].g - pi[CONV(j  ,k  ,width)].g) ;
                    diff_b = (new[CONV(j  ,k  ,width)].b - pi[CONV(j  ,k  ,width)].b) ;

                    // if(j > height/10-size && j < j_cond)
                    //     printf("diffr: %f, diffg: %f, diffb: %f, \n", diff_r, diff_g, diff_b);

                    if ( diff_r > threshold || -diff_r > threshold
                            ||
                            diff_g > threshold || -diff_g > threshold
                            ||
                            diff_b > threshold || -diff_b > threshold
                    ) {
                        end = 0 ; //*** FLAG (do while loop)
                    }
                    
                    //*** update p
                    pi[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                    pi[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                    pi[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;

                } 
                // Middle part
                else 
                {
                    new[CONV(j,k,width)].r = pi[CONV(j,k,width)].r ;
                    new[CONV(j,k,width)].g = pi[CONV(j,k,width)].g ;
                    new[CONV(j,k,width)].b = pi[CONV(j,k,width)].b ;
                }
                
            }
        }
    } while ( threshold > 0 && !end ) ;

    // printf( "Nb iter for image %d\n", n_iter ) ;int i=0; i<N; i++)

    free (new) ;

}
