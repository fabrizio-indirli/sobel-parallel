#include "blur_filter.h"

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

__global__ void compute_blur_filter(pixel* newP, pixel* pi, int height, int width, int size)
{
    int nHeight = height/10-size;
    int nWidth = width-size;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    // int j, k;
    /* Apply blur on top part of image (10%) */
    // for(j=size; j < nHeight; j++)
    if(j >= size && j < nHeight)
    {
        // for(k=size; k < nWidth; k++)
        if (k >= size && k < nWidth)
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

            newP[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
            newP[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
            newP[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
        }
    }

    /* Apply blur on the bottom part of the image (10%) */
    // for(j=height*0.9+size; j<height-size; j++)
    if(j >= height*0.9+size && j < height-size)
    {
        // for(k=size; k<width-size; k++)
        if(k >= size && k < nWidth)
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

            newP[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
            newP[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
            newP[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
        }
    }

    /* Copy the middle part of the image */
    // for(j=height/10-size; j<height*0.9+size; j++)
    if (j >= height/10-size && j < height*0.9+size)
    {
        // for(k=size; k<width-size; k++)
        if (k >= size && k < width-size)
        {
            newP[CONV(j,k,width)].r = pi[CONV(j,k,width)].r ; 
            newP[CONV(j,k,width)].g = pi[CONV(j,k,width)].g ; 
            newP[CONV(j,k,width)].b = pi[CONV(j,k,width)].b ; 
        }
    }
    
}



// void
// apply_blur_filter( int width, int height, pixel * pi, int size, int threshold ) // 5, 20
// {   /*This version of the sobel filter works only on one image at a time*/

//     int n_iter = 0 ;
//     int end = 0 ;
//     int j, k ;
//     pixel * new ;

//    /* Allocate array of new pixels */
//     new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

//     /* Perform at least one blur iteration */
//     do
//     {
//         end = 1 ;
//         n_iter++ ;

//         /* Apply blur on top part of image (10%) */
//         for(j=size; j<height/10-size; j++)
//         {
//             for(k=size; k<width-size; k++)
//             {
//                 int stencil_j, stencil_k ;
//                 int t_r = 0 ;
//                 int t_g = 0 ;
//                 int t_b = 0 ;

//                 for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
//                 {
//                     for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
//                     {
//                         t_r += pi[CONV(j+stencil_j,k+stencil_k,width)].r ;
//                         t_g += pi[CONV(j+stencil_j,k+stencil_k,width)].g ;
//                         t_b += pi[CONV(j+stencil_j,k+stencil_k,width)].b ;
//                     }
//                 }

//                 new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
//                 new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
//                 new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
//             }
//         }

//         /* Copy the middle part of the image */
//         for(j=height/10-size; j<height*0.9+size; j++)
//         {
//             for(k=size; k<width-size; k++)
//             {
//                 new[CONV(j,k,width)].r = pi[CONV(j,k,width)].r ; 
//                 new[CONV(j,k,width)].g = pi[CONV(j,k,width)].g ; 
//                 new[CONV(j,k,width)].b = pi[CONV(j,k,width)].b ; 
//             }
//         }

//         /* Apply blur on the bottom part of the image (10%) */
//         for(j=height*0.9+size; j<height-size; j++)
//         {
//             for(k=size; k<width-size; k++)
//             {
//                 int stencil_j, stencil_k ;
//                 int t_r = 0 ;
//                 int t_g = 0 ;
//                 int t_b = 0 ;

//                 for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
//                 {
//                     for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
//                     {
//                         t_r += pi[CONV(j+stencil_j,k+stencil_k,width)].r ;
//                         t_g += pi[CONV(j+stencil_j,k+stencil_k,width)].g ;
//                         t_b += pi[CONV(j+stencil_j,k+stencil_k,width)].b ;
//                     }
//                 }

//                 new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
//                 new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
//                 new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
//             }
//         }

//         for(j=1; j<height-1; j++)
//         {
//             for(k=1; k<width-1; k++)
//             {

//                 float diff_r ;
//                 float diff_g ;
//                 float diff_b ;

//                 diff_r = (new[CONV(j  ,k  ,width)].r - pi[CONV(j  ,k  ,width)].r) ;
//                 diff_g = (new[CONV(j  ,k  ,width)].g - pi[CONV(j  ,k  ,width)].g) ;
//                 diff_b = (new[CONV(j  ,k  ,width)].b - pi[CONV(j  ,k  ,width)].b) ;

//                 if ( diff_r > threshold || -diff_r > threshold 
//                         ||
//                             diff_g > threshold || -diff_g > threshold
//                             ||
//                             diff_b > threshold || -diff_b > threshold
//                     ) {
//                     end = 0 ;
//                 }

//                 pi[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
//                 pi[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
//                 pi[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
//             }
//         }

//     } while ( threshold > 0 && !end ) ;

//     // printf( "Nb iter for image %d\n", n_iter ) ;int i=0; i<N; i++)

//     free (new) ;

// }

