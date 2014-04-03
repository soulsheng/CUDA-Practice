
#include "reduction_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM	256

__global__ void reduction_kernel( float* array, int size )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if( bid*blockDim.x + tid > size )
		return;

	int offset = bid * blockDim.x;
	float* arraySub = array + offset;

	for ( int d=blockDim.x/2;d>=1; d=d/2 )
	{
		__syncthreads();
		if( tid < d )
		{
			arraySub[tid] += arraySub[tid+d];
		}
	}
}

__global__ void reduction_kernel1( float* array, int size )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if( bid*blockDim.x + tid > size )
		return;

	int offset = bid * blockDim.x;
	float* arraySub = array + offset;

	for ( int d=1;d<=blockDim.x/2; d=d*2 )
	{
		__syncthreads();
		if( tid % (2*d) == 0)
		{
			arraySub[tid] += arraySub[tid+d];
		}
	}
}

void reduction_block( float* array, int offset, int size)
{
	for(int i=0;i<size;i++)
		array[0] += array[i*offset] ;
}

float reduction_gpu( float* array, int size )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	reduction_kernel<<< countBlock, sizeBlock >>>( d_array, size );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	reduction_block( array, sizeBlock, countBlock );

	return array[0];
}

float reduction_gpu1( float* array, int size )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	reduction_kernel1<<< countBlock, sizeBlock >>>( d_array, size );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	reduction_block( array, sizeBlock, countBlock );

	return array[0];
}

void warnup_gpu( float* array, int size )
{
	reduction_gpu( array, size );
}
