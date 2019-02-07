#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define N 9

int main()
{
    int i, rank;

    struct timeval t1, t2;
    double duration ;

    gettimeofday(&t1, NULL);

 
    #pragma omp parallel default(none) private(rank,i)
    {
        rank = omp_get_thread_num();
        
        #pragma omp for schedule(static,1)
            for (i=0; i < N; ++i){
        #pragma omp ordered 
            {
                printf("Rank: %d ; iteration: %d\n", rank, i);
            }
        }
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf( "in %lf s\n", duration );

    return 0;
}