
#include "rgb2gray_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM	256
#define		R_RATIO	0.299f
#define		G_RATIO	0.587f
#define		B_RATIO	0.114f

__global__ void rgb2gray_kernel2( float* rgb, float* gray, int size )
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(  index > size )
		return;
	
	gray[ index ] = 
		rgb[ index +0] * R_RATIO 
	+	rgb[ index +size] * G_RATIO 
	+	rgb[ index +size*2] * B_RATIO  ;
}

void rgb2gray_gpu2( float* rgb, float* gray, int size )
{
	float* d_arrayA ;
	cudaMalloc( (void**)&d_arrayA, sizeof(float)*size*3 );
	cudaMemcpy( d_arrayA, rgb, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayC ;
	cudaMalloc( (void**)&d_arrayC, sizeof(float)*size );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	rgb2gray_kernel2<<< countBlock, sizeBlock >>>(  d_arrayA,  d_arrayC, size );

	cudaMemcpy( gray, d_arrayC, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_arrayA );
	cudaFree( d_arrayC );

}

__global__ void rgb2gray_kernel1( float* rgb, float* gray, int size )
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(  index > size )
		return;
	
	gray[ index ] =
		rgb[ index*3 ] * R_RATIO 
	+	rgb[ index*3 +1] * G_RATIO 
	+	rgb[ index*3 +2] * B_RATIO  ;
}

void rgb2gray_gpu1( float* rgb, float* gray, int size )
{
	float* d_arrayA ;
	cudaMalloc( (void**)&d_arrayA, sizeof(float)*size*3 );
	cudaMemcpy( d_arrayA, rgb, sizeof(float)*size*3, cudaMemcpyHostToDevice );

	float* d_arrayC ;
	cudaMalloc( (void**)&d_arrayC, sizeof(float)*size );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;
	rgb2gray_kernel1<<< countBlock, sizeBlock >>>( d_arrayA, d_arrayC, size );

	cudaMemcpy( gray, d_arrayC, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_arrayA );
	cudaFree( d_arrayC );

}

void warnup_gpu( float* rgb, float* gray, int size )
{
	rgb2gray_gpu1( rgb, gray, size );
}
