#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;


__global__ void compute_average(pixel* p, int N)
{
    // blockIdx.x * blockDim.x + threadIdx.x;
    // int i = blockIdx.x;
    // int i = blockDim.x;
    int i = threadIdx.x;
    if (i < N)
    {
        int moy;
        moy = (p[i].r + p[i].g + p[i].b)/3;
        p[i].r = moy;
        p[i].g = moy;
        p[i].b = moy;
    }
}


int main()
{
    // pixel* grayi;
    // grayi = (pixel*)malloc( N * sizeof(pixel) ); 

    int N = 2;

    pixel* p;
    p = (pixel*)malloc( N * sizeof(pixel)); // A, B (2 pixels)
    p[0].r = 16;
    p[0].g = 32;
    p[0].b = 48;

    p[1].r = 64;
    p[1].g = 128;
    p[1].b = 192;

    /* GPU */
    pixel* dp;
    cudaMalloc((void**)&dp, N * sizeof(pixel));

    cudaMemcpy(dp, p, N * sizeof(pixel), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2); // blockDim.x, blockDim.y, blockDim.z
    dim3 numBlocks(1); // blockIdx.x, blockIdx.y, blockIdx.z
    // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.x);
    printf("numBlocks (%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);

    compute_average<<<numBlocks,threadsPerBlock>>>(dp, N);

    cudaMemcpy(p, dp, N * sizeof(pixel), cudaMemcpyDeviceToHost);

    int i,j;
    printf("r, g, b\n%d %d %d\n", p[0].r, p[0].g, p[0].b);
    printf("r, g, b\n%d %d %d\n", p[1].r, p[1].g, p[1].b);


    return 0;
}