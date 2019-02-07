#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define N 9


int main()
{
    int i, j, rank;
    struct timeval t1, t2;
    double duration ;
    
    int sum;
    gettimeofday(&t1, NULL);

    #pragma omp parallel default(none) shared(i,j, sum) private(rank)
    {
        #pragma omp for collapse(2) schedule(dynamic,10)
        for(i=0; i < 10; ++i)
        {
            for (j=0; j < 20; ++j)
            {
                sum = sum + i + j;
                //printf("RANK: %d,  i: %d, j: %d\n", omp_get_thread_num(), i, j);
            }
        }
    }

 
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "in %lf s\n", duration );

    return 0;
}