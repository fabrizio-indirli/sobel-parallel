#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


int main()
{   

    int rank;
    int i;
    #pragma omp parallel default(none) private(i,rank)
    {
        #pragma omp master
        {
            printf("The number of threads : %d\n", omp_get_num_threads());
        }
        
        rank = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (i=0; i < 10; ++i)
        {
            printf("i = %d from %d\n", i, rank);
        }
    }
    


    return 0;
}

