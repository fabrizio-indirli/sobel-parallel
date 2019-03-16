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

#include "load_pixels.h"
#include "store_pixels.h"
#include "grey_filter.h"
#include "blur_filter.h"
#include "sobel_filter.h"

#define SOBELF_DEBUG 1
// Either LOGGING || LOG_FILTERS
#define LOGGING 0
#define LOG_FILTERS 1

#if LOGGING
    #define FILE_NAME "./logs_plots/write_plog3.csv"
#endif

#if LOG_FILTERS
    #define FILE_NAME "./logs_plots/filter_log.csv"
#endif

#if LOGGING || LOG_FILTERS
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

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

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

    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /*Open perfomance log file for debug*/
    #if LOGGING || LOG_FILTERS
        fOut = fopen(FILE_NAME,"a");
        if(ftell(fOut)==0) //file is empty
            #if LOGGING
                fprintf(fOut, "import_time,filters_time,export_time,");
            #else 
                fprintf(fOut, "import_time,gray_time,blur_time,sobel_time,filters_time,export_time,");
            #endif
        newRow();
    #endif

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;
    #if LOGGING || LOG_FILTERS
        appendNumToRow(duration);
    #endif

    /* FILTER Timer start */
    gettimeofday(&t0, NULL);

    /***** Start of parallelized version of filters *****/
    int i;
    int width, height ;

    pixel ** p ;

    p = image->p ;

    #if LOG_FILTERS
        gettimeofday(&t1, NULL);
    #endif
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int N = image->width[i] * image->height[i];
        
        /* GPU */
        pixel* dPi;
        cudaMalloc((void**)&dPi, N * sizeof(pixel));

        cudaMemcpy(dPi, p[i], N * sizeof(pixel), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(1024); // blockDim.x, blockDim.y, blockDim.z
        dim3 numBlocks(N / threadsPerBlock.x + 1);
        // printf("threadsPerBlock (%d,%d,%d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
        // printf("numBlocks (%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);
        // printf("numBlocks (%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);

        compute_gray_filter<<<numBlocks,threadsPerBlock>>>(dPi, N);

        cudaMemcpy(p[i], dPi, N * sizeof(pixel), cudaMemcpyDeviceToHost);

    }
    #if LOG_FILTERS
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "Gray filter done in %lf s\n", duration ) ;
        appendNumToRow(duration);
        gettimeofday(&t1, NULL);
    #endif
    int j, k ;
    int end = 1;
    int * end_d;
    int size = 5; // blur kernel size
    int threshold = 20;
    pixel * newP ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;

        int N = width * height;
        int n_iter = 0;

        // Allocate array of new pixels
        newP = (pixel *)malloc(N * sizeof( pixel ) ) ; 
        
        // GPU
        pixel* dPi;
        pixel* dNewP;

        cudaMalloc((void**)&dPi, N * sizeof( pixel ));    
        cudaMalloc((void**)&dNewP, N * sizeof( pixel ));

        cudaMalloc((int **)&end_d, 1*sizeof(int));
        cudaMemcpy(end_d, &end, sizeof(int), cudaMemcpyHostToDevice);

        // copy pixels to device
        cudaMemcpy(dPi, p[i], N * sizeof( pixel ), cudaMemcpyHostToDevice);

        // Perform at least one blur iteration
        do
        {
            dim3 threadsPerBlock(32,32); // blockDim.x, blockDim.y, blockDim.z
            dim3 numBlocks(height/32+1, width/32+1); // +1 or not...
            compute_blur_filter<<<numBlocks,threadsPerBlock>>>(dNewP, dPi, height, width, size, end_d, threshold);

            // cudaMemcpy(newP, dNewP, N * sizeof( pixel ), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(&end, end_d, sizeof(int), cudaMemcpyDeviceToHost);
            printf("On iteration %d of image %d the value of end is: %d\n", n_iter, i, end);
            n_iter++;

        }
        while ( threshold > 0 && !end ) ;

        // copy pixels back to ram
        cudaMemcpy(p[i], dPi, N * sizeof(pixel), cudaMemcpyDeviceToHost);

    }
    #if LOG_FILTERS
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "Blur filter done in %lf s\n", duration ) ;
        appendNumToRow(duration);
        gettimeofday(&t1, NULL);
    #endif
    //#pragma omp parallel for shared(p) schedule(dynamic)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;
        pixel * pi = p[i]; // input

        // /* CUDA ver. */
        // int N = width*height;

        // pixel* sobel;

        // sobel = (pixel *)malloc(N * sizeof( pixel ) ) ; // new image

        // /* GPU */
        // pixel* dPi;
        // pixel* dSobel;
 
        // cudaMalloc((void**)&dPi, N * sizeof( pixel ));
        // cudaMalloc((void**)&dSobel, N * sizeof( pixel ));
        // // malloc inside CUDA kernel?? possible? Not really used in CPU side...
 
        // cudaMemcpy(dPi, p[i], N * sizeof( pixel ), cudaMemcpyHostToDevice);
 
        // dim3 threadsPerBlock(32,32); // blockDim.x, blockDim.y, blockDim.z
        // dim3 numBlocks(height/32+1, width/32+1); // +1 or not...
        // compute_sobel_filter<<<numBlocks,threadsPerBlock>>>(dSobel, dPi, height, width);
 
        // // cudaMemcpy(sobel, dSobel, N * sizeof( pixel ), cudaMemcpyDeviceToHost);
        // cudaMemcpy(sobel, dSobel, N * sizeof( pixel ), cudaMemcpyDeviceToHost);

        // for(j=1; j<height-1; j++)
        // {
        //     for(k=1; k<width-1; k++)
        //     {
        //         p[i][CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
        //         p[i][CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
        //         p[i][CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
        //     }
        // }

        // free (sobel) ;

        /* (CPU ver.) Apply sobel filter on pixels */
        apply_sobel_filter(width, height, pi);

    }
    #if LOG_FILTERS
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "Sobel filter done in %lf s\n", duration ) ;
        appendNumToRow(duration);
    #endif
    /***** End of parallelized version of filters *****/

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t0.tv_sec)+((t2.tv_usec-t0.tv_usec)/1e6);
    printf( "All filters done in %lf s\n", duration ) ;
    #if LOGGING || LOG_FILTERS
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
    #if LOGGING || LOG_FILTERS
        appendNumToRow(duration);
    #endif

    /*Close perfomance log file*/
    #if LOGGING || LOG_FILTERS
        fclose(fOut);
    #endif

    return 0 ;
}
