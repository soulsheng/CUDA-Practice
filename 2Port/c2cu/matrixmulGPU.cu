
#include <cuda_runtime.h>

#include <iostream>
using namespace std;

__global__ void kernelMatrixMul( float* a, float*b, float*c, int n )
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if( ii >= n*n )
		return;

	int i= ii / n ;
	int j= ii %n;

	for(int k=0;k<n;k++)
		c[ i*n +j] += a[ i*n + k] * b[ k*n +j] ;
}

extern "C" void matrixMulGPU( float* a, float*b, float*c, int n )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );

	int nBlock = 256;
	int nGrid = (n*n + nBlock-1)/nBlock;

	kernelMatrixMul<<< nGrid, nBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );
}