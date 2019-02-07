#include "grey_filter.h"

void
apply_gray_filter( int width, int height, pixel * pi )
{
    /*This version of the grey filter works only on one image at a time*/
    int j;

    /* #if SOBELF_DEBUG
        struct timeval t0, t1, t2, tf; //added for time checking
        double duration; //added for time checking
        gettimeofday(&t1, NULL);
    #endif */

    
    for ( j = 0 ; j < width * height ; j++ )
    {
        int moy ;

        // moy = pi[j].r/4 + ( pi[j].g * 3/4 ) ;
        moy = (pi[j].r + pi[j].g + pi[j].b)/3 ;
        if ( moy < 0 ) moy = 0 ;
        if ( moy > 255 ) moy = 255 ;

        pi[j].r = moy ;
        pi[j].g = moy ;
        pi[j].b = moy ;
    }
    
    /* #if SOBELF_DEBUG
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf( "[DEBUG] Time needed to apply grey filter to all the images: %lf s\n",  duration);
    #endif */
}

