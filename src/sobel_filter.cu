#include "sobel_filter.h"
#include <stdio.h>

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)


__global__ void kernel_sobel_filter(pixel* sobel, pixel* pi, int height, int width)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    // for(j=1; j<height-1; j++)
    if(j >= 1 && j < height-1)
    {
        // for(k=1; k<width-1; k++)
        if(k >= 1 && k < width-1)
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

}


// applies sobel filter on one image, without OpenMP
void apply_sobel_filter(int width, int height, pixel * pi){
    /*This version of the sobel filter works only on one image at a time*/
    int j, k;
    pixel * sobel ;
    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

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

    
// applies sobel filter on one image with OpenMP
void apply_sobel_filter_omp(int width, int height, pixel * pi){
    /*This version of the sobel filter works only on one image at a time*/
    int j, k;
    pixel * sobel ;
    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    #pragma omp parallel default(none) private(j,k) shared(width,height,pi,sobel) //***
    {
        // `dynamic` can be a better choice, since there is an if statement that might invoke imbalance for the iteration.
        // Actually nope... static one is faster. 
        #pragma omp for collapse(2) schedule(static) 
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

        #pragma omp for collapse(2) schedule(static)
        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                pi[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                pi[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                pi[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }
    }
    free(sobel) ;
} 


// applies sobel filter on one part of one image, with OpenMP
void apply_sobel_filter_part(int width, int height, pixel * pi, int startheight, int finalheight){
    /*This version of the sobel filter works only on one image at a time*/
    int j, k;
    pixel * sobel ;
    sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

    if(finalheight==height) finalheight--;
    
    #pragma omp parallel default(none) private(j,k) shared(width,height,pi,sobel,startheight,finalheight) //***
    {   
        #pragma omp for collapse(2) schedule(static)
        for(j=1; j<finalheight; j++)
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

            #pragma omp for collapse(2) schedule(static)
            for(j=1; j<finalheight; j++)
            {
                for(k=1; k<width-1; k++)
                {
                    pi[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                    pi[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                    pi[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
                }
            }
        }
        free (sobel) ;

}





// applies sobel filter using either CUDA or OpenMP (on single picture)
void sobel_filter_auto(int width, int height, pixel * pi){
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    printf("Num of GPUs: %d\n", nDevices);

    int j, k;

    if(nDevices > 0){
        // use CUDA if GPU is available
        printf("Computing SOBEL using CUDA\n");
        /* CUDA ver. */
        int N = width*height;

        pixel* sobel;

        sobel = (pixel *)malloc(N * sizeof( pixel ) ) ; // new image

        /* GPU */
        pixel* dPi;
        pixel* dSobel;
 
        cudaMalloc((void**)&dPi, N * sizeof( pixel ));
        cudaMalloc((void**)&dSobel, N * sizeof( pixel ));
 
        cudaMemcpy(dPi, pi, N * sizeof( pixel ), cudaMemcpyHostToDevice);
 
        dim3 threadsPerBlock(32,32); // blockDim.x, blockDim.y, blockDim.z
        dim3 numBlocks(height/32+1, width/32+1); // +1 or not...
        kernel_sobel_filter<<<numBlocks,threadsPerBlock>>>(dSobel, dPi, height, width);
 
        cudaMemcpy(sobel, dSobel, N * sizeof( pixel ), cudaMemcpyDeviceToHost);

        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                pi[CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                pi[CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                pi[CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }

        free (sobel) ;

    }
}