
#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#include "matrixmulGPU.cuh"

__global__ void kernelMatrixMul2( float* a, float*b, float*c, int n )
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if( ii >= n*n )
		return;

	int y= ii / n ;
	int x= ii %n;

	for(int k=0;k<n;k++)
		c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

__global__ void kernelMatrixMul1( float* a, float*b, float*c, int n )
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if( x >= n || y >=n )
		return;

	for(int k=0;k<n;k++)
		c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

__global__ void kernelMatrixMul3( float* a, float*b, float*c, int n )
{
}

void matrixMulGPU2( float* a, float*b, float*c, int n )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 256;
	int nGrid = (n*n + nBlock-1)/nBlock;

	kernelMatrixMul2<<< nGrid, nBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );

}

void matrixMulGPU1( float* a, float*b, float*c, int n )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 16;
	int nGrid = (n + nBlock-1)/nBlock;
	dim3 sizeBlock(nBlock, nBlock);
	dim3 sizeGrid( nGrid, nGrid );

	kernelMatrixMul1<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}

void matrixMulGPU3( float* a, float*b, float*c, int n )
{
}