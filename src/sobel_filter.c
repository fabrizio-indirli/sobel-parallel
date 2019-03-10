#include "sobel_filter.h"

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)


void apply_sobel_filter(int width, int height, pixel * pi){
    /*This version of the sobel filter works only on one image at a time*/
    int j, k;
    pixel * sobel ;
    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    // `dynamic` can be a better choice, since there is an if statement that might invoke imbalance for the iteration.
    // Actually nope... static one is faster. 
    for(j=1; j<height-1; j++)
    {
        for(k=1; k<width-1; k++)
        {
            int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_so, pixel_blue_s, pixel_blue_se;
            int pixel_blue_o , pixel_blue  , pixel_blue_e ;

            float deltaX_blue ;
            float deltaY_blue ;
            float val_blue;

            pixel_blue_no = pi[CONV(j-1,k-1,width)].b ;
            pixel_blue_n  = pi[CONV(j-1,k  ,width)].b ;
            pixel_blue_ne = pi[CONV(j-1,k+1,width)].b ;
            pixel_blue_so = pi[CONV(j+1,k-1,width)].b ;
            pixel_blue_s  = pi[CONV(j+1,k  ,width)].b ;
            pixel_blue_se = pi[CONV(j+1,k+1,width)].b ;
            pixel_blue_o  = pi[CONV(j  ,k-1,width)].b ;
            pixel_blue    = pi[CONV(j  ,k  ,width)].b ;
            pixel_blue_e  = pi[CONV(j  ,k+1,width)].b ;

            deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;

            deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

            val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


            if ( val_blue > 50 )
            {
                sobel[CONV(j  ,k  ,width)].r = 255 ;
                sobel[CONV(j  ,k  ,width)].g = 255 ;
                sobel[CONV(j  ,k  ,width)].b = 255 ;
            } else
            {
                sobel[CONV(j  ,k  ,width)].r = 0 ;
                sobel[CONV(j  ,k  ,width)].g = 0 ;
                sobel[CONV(j  ,k  ,width)].b = 0 ;
            }
        }
    }

    for(j=1; j<height-1; j++)
    {
        for(k=1; k<width-1; k++)
        {
            pi[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
            pi[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
            pi[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
        }
    }
    free(sobel) ;
} 

