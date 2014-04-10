
#include "scan_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM_MAX	(1<<10)

__global__ void scan_kernel1( float* array, int size, int width )
{
	int row = threadIdx.x;

	int WIDTH = size/BLOCKDIM_MAX ;

	for (int i=1;i<WIDTH;i++)
	{
		array[i + row*WIDTH] += array[i-1 + row*WIDTH];
	}

}

float scan_gpu1( float* array, int size, int width )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size/BLOCKDIM_MAX > BLOCKDIM_MAX?BLOCKDIM_MAX: size/BLOCKDIM_MAX;
	int countBlock = 1;
	scan_kernel1<<< 1, sizeBlock >>>( d_array, size, width );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return array[size-1];
}

__global__ void scan_kernel2( float* array, int size, int width )
{
	int index = blockIdx.x * width + threadIdx.x;

	if( index > size )
		return;

	__shared__ float sdata[BLOCKDIM_MAX*2];

	sdata[threadIdx.x] = array[index];
	__syncthreads();

	int first = 0;

	for ( int d=1;d<=blockDim.x/2; d+=d, first=blockDim.x-first )
	{
		if( threadIdx.x < d )
			sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first];
		else
			sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first] + sdata[threadIdx.x-d+first];
		__syncthreads();
	}

	array[index] = sdata[threadIdx.x+first];
	__syncthreads();
}

float scan_gpu2( float* array, int size, int width )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = width>BLOCKDIM_MAX?BLOCKDIM_MAX: width;
	int countBlock = size / width ;// 行数，一个block处理一行
	scan_kernel2<<< countBlock, sizeBlock >>>( d_array, size, width );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return array[size-1];
}

void warnup_gpu( float* array, int size, int width )
{
	scan_gpu2( array, size, width );
}
