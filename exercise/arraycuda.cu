#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// __global__ void compute_gray_filter( pixel* grayi, pixel* pi, int N )
// {
//     int moy;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     for ( j=0; j < N; ++j )
//     {
//         moy = (pi[j].r + pi[j].g + pi[j].b)/3 ;
//         if ( moy < 0 ) moy = 0 ;
//         if ( moy > 255 ) moy = 255 ;
//         gray[j] = moy;
//     }
// }

__global__ void compute_addition(int* C, int* A, int* B, int N)
{
    int i;
    for (i=0; i < N; i++)
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

    cudaMalloc((void**)&dA, N * sizeof(int));
    cudaMalloc((void**)&dB, N * sizeof(int));
    cudaMalloc((void**)&dC, N * sizeof(int));

    cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(int), cudaMemcpyHostToDevice);

    compute_addition<<<1,256>>>(dC, dA, dB, N);

    cudaMemcpy(A, dC, N * sizeof(int), cudaMemcpyDeviceToHost);

    // /* GPU */
    // pixel* dGrayi;
    // pixel* dPi;

    // cudaMalloc((void**)&dGrayi, N * sizeof(pixel));
    // cudaMalloc((void**)&dPi, N * sizeof(pixel));

    // cudaMemcpy(dPi, p[i], N * sizeof(pixel), cudaMemcpyHostToDevice);



    // cudaMemcpy(grayi, dgrayi, N * sizeof(pixel), cudaMemcpyDeviceToHost);

    // /* GPU */
    // pixel* dPi;
    // int* dGray; // Output 

    // cudaMalloc((void**)&dPi, N * sizeof(pixel));
    // cudaMalloc((void**)&dGray, N * sizeof(int));

    // cudaMemcpy(dPi, p[i], N * sizeof(pixel), cudaMemcpyHostToDevice);

    // compute_gray_filter<<<1,256>>>(dGray, dPi, N);

    // cudaMemcpy(gray, dGray, N * sizeof(int), cudaMemcpyDeviceToHost);
    int i;
    for (i=0; i < 6; ++i)
    {
        printf("%d\n", A[i]);
    }



    return 0;
}