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
    //#pragma omp parallel for shared(p) schedule(dynamic)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;
        pixel * pi = p[i];

        /*Apply grey filter: convert the pixels into grayscale */
        apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        // apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        // apply_sobel_filter(width, height, pi);

    }
    #if LOG_FILTERS
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "Gray filter done in %lf s\n", duration ) ;
        appendNumToRow(duration);
        gettimeofday(&t1, NULL);
    #endif
    //#pragma omp parallel for shared(p) schedule(dynamic)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;
        pixel * pi = p[i];

        /*Apply grey filter: convert the pixels into grayscale */
        // apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
        // apply_sobel_filter(width, height, pi);

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
        pixel * pi = p[i];

        /*Apply grey filter: convert the pixels into grayscale */
        // apply_gray_filter(width, height, pi);

        /*Apply blur filter with convergence value*/
        // apply_blur_filter( width, height, pi, 5, 20 ) ;

        /* Apply sobel filter on pixels */
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
