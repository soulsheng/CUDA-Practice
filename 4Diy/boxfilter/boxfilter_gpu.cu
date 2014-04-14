
#include "boxfilter_cpu.h"
#include "timerCUDA.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#define  BLOCKDIM_MAX	(1<<10)

__global__ void boxfilter_kernel_accumulate1s( float* array, float* arrayTemp, int size, int width, int r )
{
	int row = threadIdx.x;

	for (int i=0;i<width;i++)
	{
		if(i-r>=0 && i+r<=width-1)
		{
			for(int j=i-r;j<=i+r;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
		else if(i-r<0)
		{
			for(int j=0;j<=i+r;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
		else
		{
			for(int j=i-r;j<=width-1;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
	}
}

float boxfilter_gpu1s( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );
	float* d_arrayTemp ;
	cudaMalloc( (void**)&d_arrayTemp, sizeof(float)*size );

	cudaMemcpy( d_arrayTemp, array, sizeof(float)*size, cudaMemcpyHostToDevice );
	cudaMemset( d_array, 0, sizeof(float)*size );

	int sizeBlock = size/width > BLOCKDIM_MAX?BLOCKDIM_MAX: size/width;
	int countBlock = 1;

	timerCUDA timer;
	timer.start();

	boxfilter_kernel_accumulate1s<<< countBlock, sizeBlock >>>( d_array, d_arrayTemp, size, width, r );

	timer.stop();

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );
	cudaFree( d_arrayTemp );

	return timer.getTime();
}


__global__ void boxfilter_kernel_accumulate1p( float* array, float* arrayTemp, int size, int width, int r )
{
	int i = threadIdx.x;
	int row = blockIdx.x;

	//for (int i=0;i<width;i++)
	{
		if(i-r>=0 && i+r<=width-1)
		{
			for(int j=i-r;j<=i+r;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
		else if(i-r<0)
		{
			for(int j=0;j<=i+r;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
		else
		{
			for(int j=i-r;j<=width-1;j++)
				array[i+row*width] += arrayTemp[row*width+j];
		}
	}
}

float boxfilter_gpu1p( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );
	float* d_arrayTemp ;
	cudaMalloc( (void**)&d_arrayTemp, sizeof(float)*size );

	cudaMemcpy( d_arrayTemp, array, sizeof(float)*size, cudaMemcpyHostToDevice );
	cudaMemset( d_array, 0, sizeof(float)*size );

	int sizeBlock = width > BLOCKDIM_MAX?BLOCKDIM_MAX: width;
	int countBlock = size/width;

	timerCUDA timer;
	timer.start();

	boxfilter_kernel_accumulate1p<<< countBlock, sizeBlock >>>( d_array, d_arrayTemp, size, width, r );

	timer.stop();

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );
	cudaFree( d_arrayTemp );

	return timer.getTime();
}

__global__ void boxfilter_kernel_scan2s( float* array, int size, int width )
{
	for (int i=1;i<width;i++)
	{
		array[i + threadIdx.x*width] += array[i-1 + threadIdx.x*width];
	}

}

__global__ void boxfilter_kernel_delta2s( float* array, float* arrayBackup, int size, int width, int r )
{
	int nRight=0, nLeft=0;
	int row=threadIdx.x;

	for (int i=0;i<width;i++)
	{
		if(i<=r)
		{
			nLeft = 0;
			nRight = i+r;
			array[i+row*width] = arrayBackup[row*width + nRight] ;
		}
		else if( i>r && i<width-r )
		{
			nLeft = i-r-1;
			nRight = i+r;
			array[i+row*width] = arrayBackup[row*width + nRight] - arrayBackup[row*width + nLeft];
		}
		else//if( i>width-r && i<width )
		{
			nLeft = i-r-1;
			nRight = width-1;
			array[i+row*width] = arrayBackup[row*width + nRight] - arrayBackup[row*width + nLeft];
		}
	}


}

float boxfilter_gpu2s( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size/width > BLOCKDIM_MAX?BLOCKDIM_MAX: size/width;
	int countBlock = 1;

	timerCUDA timer;
	timer.start();

	// 第一步，扫描累加
	boxfilter_kernel_scan2s<<< countBlock, sizeBlock >>>( d_array, size, width );
#if 1
	// 第二步，等间隔相减
	float* d_arrayBackup ;
	cudaMalloc( (void**)&d_arrayBackup, sizeof(float)*size );
	cudaMemcpy( d_arrayBackup, d_array, sizeof(float)*size, cudaMemcpyDeviceToDevice );
	boxfilter_kernel_delta2s<<< 1, sizeBlock >>>( d_array, d_arrayBackup, size, width, r );
#endif
	timer.stop();

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return timer.getTime();
}

__global__ void boxfilter_kernel2p( float* array, int size, int width, int r )
{
	int index = blockIdx.x * width + threadIdx.x;

	if( index > size )
		return;

	__shared__ float sdata[BLOCKDIM_MAX*2];
	// 第一步，扫描累加
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

	// 第二步，等间隔相减
	int i=threadIdx.x;

	if(threadIdx.x<=r)
	{
		sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first + r] ;
	}
	else if( threadIdx.x>r && threadIdx.x<width-r )
	{
		sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first + r] - sdata[threadIdx.x+first -r-1];
	}
	else//if( threadIdx.x>width-r && threadIdx.x<width )
	{
		sdata[threadIdx.x+blockDim.x-first] = sdata[first + width-1] - sdata[threadIdx.x+first -r-1];
	}

	array[index] = sdata[threadIdx.x+blockDim.x-first];
	__syncthreads();
}

float boxfilter_gpu2p( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	timerCUDA timer;
	timer.start();
	int sizeBlock = width>BLOCKDIM_MAX?BLOCKDIM_MAX: width;
	int countBlock = size / width ;// 行数，一个block处理一行
	boxfilter_kernel2p<<< countBlock, sizeBlock >>>( d_array, size, width, r );
	timer.stop();

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return timer.getTime();
}

void warnup_gpu( float* array, int size, int width, int r )
{
	boxfilter_gpu2p( array, size, width, r );
}
