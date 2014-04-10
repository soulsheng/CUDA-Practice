
#include "vectorDot_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM	256


__global__ void vectorDot_kernel2( float* arrayA, float* arrayB, float* arrayC, int size )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int index = bid*blockDim.x + tid;

	if(  index > size )
		return;
	
	arrayC[ index ] =    arrayA[ index +0] * arrayB[ index +0] 
	+ arrayA[ index +size] * arrayB[ index +size] 
	+ arrayA[ index +size*2] * arrayB[ index +size*2]  ;
}

void vectorDot_gpu2( float* arrayA, float* arrayB, float* arrayC, int size )
{
	float* d_arrayA ;
	cudaMalloc( (void**)&d_arrayA, sizeof(float)*size*3 );

	cudaMemcpy( d_arrayA, arrayA, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayB ;
	cudaMalloc( (void**)&d_arrayB, sizeof(float)*size*3 );

	cudaMemcpy( d_arrayB, arrayB, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayC ;
	cudaMalloc( (void**)&d_arrayC, sizeof(float)*size );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	vectorDot_kernel2<<< countBlock, sizeBlock >>>(  d_arrayA,  d_arrayB, d_arrayC, size );

	cudaMemcpy( arrayC, d_arrayC, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_arrayA );
	cudaFree( d_arrayB );
	cudaFree( d_arrayC );

}

__global__ void vectorDot_kernel1( float* arrayA, float* arrayB, float* arrayC, int size )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int index = bid*blockDim.x + tid;

	if(  index > size )
		return;
	
	arrayC[ index ] = 
		arrayA[ index*3 ] * arrayB[ index*3 ] 
	+ arrayA[ index*3 +1] * arrayB[ index*3 +1] 
	+ arrayA[ index*3 +2] * arrayB[ index*3 +2]  ;
}

void vectorDot_gpu1( float* arrayA, float* arrayB, float* arrayC, int size )
{
	float* d_arrayA ;
	cudaMalloc( (void**)&d_arrayA, sizeof(float)*size*3 );

	cudaMemcpy( d_arrayA, arrayA, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayB ;
	cudaMalloc( (void**)&d_arrayB, sizeof(float)*size*3 );

	cudaMemcpy( d_arrayB, arrayB, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayC ;
	cudaMalloc( (void**)&d_arrayC, sizeof(float)*size );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	vectorDot_kernel1<<< countBlock, sizeBlock >>>( d_arrayA, d_arrayB, d_arrayC, size );

	cudaMemcpy( arrayC, d_arrayC, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_arrayA );
	cudaFree( d_arrayB );
	cudaFree( d_arrayC );

}

void warnup_gpu( float* arrayA, float* arrayB, float* arrayC, int size )
{
	vectorDot_gpu1( arrayA, arrayB, arrayC, size );
}
