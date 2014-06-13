
#include "reduction_cpu.h"
#include "timerTest.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#define  BLOCKDIM	256

// GPU 版本3：shared memory
__global__ void reduction_kernel3( float* array, int size )
{
	__shared__ float sarray[BLOCKDIM];

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if( bid*blockDim.x + tid > size )
		return;

	int offset = bid * blockDim.x;
	float* arraySub = array + offset;

	sarray[tid] = arraySub[tid];
	__syncthreads();

	for ( int d=blockDim.x/2;d>=1; d=d/2 )
	{
		__syncthreads();
		if( tid < d )
		{
			sarray[tid] += sarray[tid+d];
		}
	}

	arraySub[tid] = sarray[tid];
	__syncthreads();

}

// GPU 版本2：跨步长累加，步长递减
__global__ void reduction_kernel2( float* array, int size )
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

// GPU 版本1：跨步长累加，步长递增
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

float reduction_gpu3( float* array, int size, bool bTimeKernel )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;

	timerTestCU	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	reduction_kernel3<<< countBlock, sizeBlock >>>( d_array, size );

	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	reduction_block( array, sizeBlock, countBlock );

	return array[0];
}

float reduction_gpu2( float* array, int size, bool bTimeKernel )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;

	
	timerTestCU	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	reduction_kernel2<<< countBlock, sizeBlock >>>( d_array, size );

	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	reduction_block( array, sizeBlock, countBlock );

	return array[0];
}

float reduction_gpu1( float* array, int size, bool bTimeKernel )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size>BLOCKDIM?BLOCKDIM: size;
	int countBlock = (size+ sizeBlock-1)/sizeBlock;

	
	timerTestCU	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	reduction_kernel1<<< countBlock, sizeBlock >>>( d_array, size );

	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	reduction_block( array, sizeBlock, countBlock );

	return array[0];
}

void warnup_gpu( float* array, int size )
{
	reduction_gpu1( array, size, false );
}
