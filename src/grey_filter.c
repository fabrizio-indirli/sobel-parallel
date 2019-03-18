#include "grey_filter.h"

// applies grey filter on the image, using OpenMP
void
apply_gray_filter_omp( int width, int height, pixel * pi )
{
    /*This version of the grey filter works only on one image at a time*/
    int j;

    #pragma omp parallel default(none) private(j) shared(pi,width,height)
    {   
        #pragma omp for schedule(static) //*** default chunk size
        for ( j = 0 ; j < width * height ; j++ )
        {
            int moy ;

            moy = (pi[j].r + pi[j].g + pi[j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            pi[j].r = moy ;
            pi[j].g = moy ;
            pi[j].b = moy ;
        }
    }
    
}



// applies the grey filter on the lines from startheight to finalheight, using also OpenMP
void
apply_gray_filter_part( int width, int height, pixel * pi, int startheight, int finalheight )
{
    /*This version of the grey filter works only on one part of one image at a time*/
    int j;



    #pragma omp parallel default(none) private(j) shared(pi,width,height, startheight, finalheight)
    {   
        #pragma omp for schedule(static) //*** default chunk size
        for ( j = width*startheight ; j < width * finalheight ; j++ )
        {
            int moy ;

            moy = (pi[j].r + pi[j].g + pi[j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            pi[j].r = moy ;
            pi[j].g = moy ;
            pi[j].b = moy ;
        }
    }
    
}


// applies grey filter on one image, without OpenMP
void
apply_gray_filter( int width, int height, pixel * pi )
{
    /*This version of the grey filter works only on one image at a time*/
    int j;
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
}

