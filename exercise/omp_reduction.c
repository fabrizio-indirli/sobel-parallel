#include <stdio.h>
#include <omp.h>

int main()
{
    int end=0;
    int n_iter=0;
    int when=0;
    do{
        end = 1 ;
        n_iter++ ;
        #pragma omp parallel 
        {   
            int rank;
            rank = omp_get_thread_num();
            #pragma omp for reduction(||:end)
            for (int i=0; i < 10; ++i)
            {
                printf("rank : %d, i : %d\n", rank, i);
                if (rank==1)
                    end=1;
                else
                    end=0;
                when += i;
            }
        }
    } while (!end);

    printf("n_iter = %d\n", n_iter);
    printf("when = %d\n", when);

    return 0;
}