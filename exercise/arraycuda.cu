#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// __global__ void compute_gray_filter( pixel* grayi, pixel* pi, int N )
// {
//     int moy;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if ( j < N )
//     {
//         moy = (pi[j].r + pi[j].g + pi[j].b)/3 ;
//         if ( moy < 0 ) moy = 0 ;
//         if ( moy > 255 ) moy = 255 ;
//         gray[j] = moy;
//     }
// }

__global__ void compute_addition(int* C, int* A, int* B, int N)
{
    // blockIdx.x * blockDim.x + threadIdx.x;
    // int i = blockIdx.x;
    // int i = blockDim.x;
    int i = threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}


int main()
{
    // pixel* grayi;
    // grayi = (pixel*)malloc( N * sizeof(pixel) ); 


    int A[] = {1,2,3,4,5,6};
    int B[6] = {6,5,4,3,2,1};
    int N = 6;

    int* C;
    C = (int*)malloc(N * sizeof(int));
    
    int* dA;
    int* dB;
    int* dC;

    int i;
    for (i=0; i < 6; ++i)
    {
        printf("%d\n", A[i]);
    }
    printf("\n", A[i]); // {1,2,3,4,5,6};

    cudaMalloc((void**)&dA, N * sizeof(int));
    cudaMalloc((void**)&dB, N * sizeof(int));
    cudaMalloc((void**)&dC, N * sizeof(int));

    cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // int numBlocks = 1;
    dim3 threadsPerBlock(2); // blockDim.x, blockDim.y, blockDim.z
    dim3 numBlocks(1); // blockIdx.x, blockIdx.y, blockIdx.z
    // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.x);
    printf("numBlocks (%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);

    compute_addition<<<numBlocks,threadsPerBlock>>>(dC, dA, dB, N);

    cudaMemcpy(A, dC, N * sizeof(int), cudaMemcpyDeviceToHost);
    /// A[] = {7,7,7}

    for (i=0; i < 6; ++i)
    {
        printf("%d\n", A[i]);
    }
    printf("\n", A[i]); // {0,0,0,0,0,0};
    int j;
    for (j=0; j < 3; ++j)
    {
        cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
        compute_addition<<<numBlocks,threadsPerBlock>>>(dC, dA, dB, N);
        cudaMemcpy(A, dC, N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (i=0; i < 6; ++i)
    {
        printf("%d\n", A[i]); // {0,15,0,0,0,0};
    }



    return 0;
}