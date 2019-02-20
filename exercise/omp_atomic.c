#include <stdio.h>
#include <omp.h>

int main()
{
    int c, r;
    c = 91680;
    #pragma omp parallel private(r)
    {
        r = omp_get_thread_num();
        #pragma omp atomic
            c++;
        printf("R: %d ; C : %d\n", r, c);


    }
    printf("After region, c : %d\n", c);

    return 0;
}