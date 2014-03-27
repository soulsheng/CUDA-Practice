
#include "cuda_runtime.h"

__global__ void kernel( float* aDev, float* bDev, float* cDev, int n )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index >= n*n )
		return;

	int i = index/n;
	int j = index%n;
	for (int k=0;k<n;k++)
	{
		cDev[i*n+j] += aDev[i*n+k] * bDev[k*n+j];
	}
}

__global__ void kernel2( float* aDev, float* bDev, float* cDev, int n )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index >= n )
		return;

	int i = index;
	for(int j=0;j<n;j++)
		for (int k=0;k<n;k++)
		{
			cDev[i*n+j] += aDev[i*n+k] * bDev[k*n+j];
		}
}

extern "C" 
int  multiplymatrixGPU( float* a, float* b, float* c, int n )
{
	float *aDev, *bDev, *cDev;
	cudaMalloc( (void **)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void **)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void **)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int sizeBlock = 256;
	//int sizeGrid = (n*n+sizeBlock-1)/sizeBlock;
	//kernel<<< sizeBlock, sizeGrid >>>(aDev, bDev, cDev, n);

	int sizeGrid2 = (n+sizeBlock-1)/sizeBlock;
	kernel2<<< sizeBlock, sizeGrid2 >>>(aDev, bDev, cDev, n);

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );

	return 0;
}