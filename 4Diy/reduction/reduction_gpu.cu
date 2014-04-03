
#include "reduction_cpu.h"

#include <cuda_runtime.h>

__global__ void reduction_kernel( unsigned int* array, int size )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if( bid*blockDim.x + tid > size )
		return;

	int offset = bid * blockDim.x;
	unsigned int* arraySub = array + offset;

	for ( int d=1;d<=blockDim.x/2; d=d*2 )
	{
		__syncthreads();
		if( tid % (2*d) == 0)
		{
			arraySub[tid] += arraySub[tid+d];
		}
	}
#if 0
	__syncthreads();

	if(bid==0 && tid==0&& size>blockDim.x)
	{
		for(int i=blockDim.x;i<size;i+=blockDim.x)
			arraySub[0] += arraySub[i];
	}
#endif
}

unsigned int reduction_gpu( unsigned int* array, int size )
{
	unsigned int* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(unsigned int)*size );

	cudaMemcpy( d_array, array, sizeof(unsigned int)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>256?256: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	reduction_kernel<<< countBlock, sizeBlock >>>( d_array, size );

	cudaMemcpy( array, d_array, sizeof(unsigned int)*size, cudaMemcpyDeviceToHost );

	return array[0];
}

void warnup_gpu( unsigned int* array, int size )
{
	reduction_gpu( array, size );
}
