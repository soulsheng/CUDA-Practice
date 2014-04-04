
#include "scan_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM	256

__global__ void scan_kernel( float* array, int size )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if( index > size )
		return;

	__shared__ float sdata[BLOCKDIM*2];

	sdata[threadIdx.x] = array[index];
	__syncthreads();

	int first = 0;

	for ( int d=1;d<=blockDim.x/2; d+=d, first=BLOCKDIM-first )
	{
		if( threadIdx.x < d )
			sdata[threadIdx.x+BLOCKDIM-first] = sdata[threadIdx.x+first];
		else
			sdata[threadIdx.x+BLOCKDIM-first] = sdata[threadIdx.x+first] + sdata[threadIdx.x-d+first];
		__syncthreads();
	}

	array[index] = sdata[threadIdx.x+first];
	__syncthreads();
}

float scan_gpu( float* array, int size )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	scan_kernel<<< countBlock, sizeBlock >>>( d_array, size );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return array[size-1];
}

void warnup_gpu( float* array, int size )
{
	scan_gpu( array, size );
}
