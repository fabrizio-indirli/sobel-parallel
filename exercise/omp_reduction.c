#include <stdio.h>
#include <omp.h>

int main()
{
    // int found=-1;
    // int r;
    // int found_flag = 1;
    // #pragma omp parallel private(r)
    // for (int i=0; i < 8; ++i)
    // {
    //     int f = i;
    //     if (i==7)
    //         found_flag = (found_flag & 0);
    // }
    // printf("After region, found_flag : %d\n", found_flag);
    printf("& : %d", (1 && -1 && 100 && 9000));

    return 0;
}