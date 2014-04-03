
#include "reduction_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM	128

__global__ void reduction_kernel( float* array, int size )
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
#if 1
	__syncthreads();

	if(bid==0 && tid==0&& size>blockDim.x)
	{
		for(int i=blockDim.x;i<size;i+=blockDim.x)
			arraySub[0] += arraySub[i];
	}
#endif
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

	return array[0];
}

void warnup_gpu( float* array, int size )
{
	reduction_gpu( array, size );
}
