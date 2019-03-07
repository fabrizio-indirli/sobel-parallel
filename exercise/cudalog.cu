
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void cudalog(int* I)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    // if (i == 2147483646*2) // max blockIdx.x = 2147483647
    //     I[0] = i;
    // if (j == 1)
    //     I[1] = j;
    if (j==0)
        I[j] = 10;
    else if (j==1)
        I[j] = -10;
    
}



int main()
{
    int* I = (int*)malloc(2*sizeof(int));
    int* dI;

    cudaMalloc((void**)&dI, 2*sizeof(int));

    // int numBlocks = 1;
    // 69682912 maximum image size. 
    dim3 threadsPerBlock(2); // blockDim.x, blockDim.y, blockDim.z
    // dim3 numBlocks(2147483647); // blockIdx.x, blockIdx.y, blockIdx.z
    dim3 numBlocks(0);

    /* Maximum grid size */
    // Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535) // 32bit. [-2147483648, +2147483647]
    // 2147483647 when threadsPerBlock==1

    // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.x);
    printf("numBlocks (%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);

    cudalog<<<numBlocks,threadsPerBlock>>>(dI);

    cudaMemcpy(I, dI, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("I[0] = %d\n", I[0]);
    printf("I[1] = %d\n", I[1]);
    /// A[] = {7,7,7}



    return 0;
}