
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>



int main()
{
    int j;

    #pragma omp parallel default(none) private(j)
    {
        #pragma omp for schedule(static)
        for (j=0; j <100; ++j)
        {
            printf("iter: %d from %d\n", j, omp_get_thread_num());
        }
    }

}