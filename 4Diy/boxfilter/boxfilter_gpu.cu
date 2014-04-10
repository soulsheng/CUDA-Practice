
#include "boxfilter_cpu.h"

#include <cuda_runtime.h>

#define  BLOCKDIM_MAX	(1<<10)

__global__ void boxfilter_kernel_scan1( float* array, int size, int width )
{
	for (int i=1;i<width;i++)
	{
		array[i + threadIdx.x*width] += array[i-1 + threadIdx.x*width];
	}

}

__global__ void boxfilter_kernel_delta1( float* array, float* arrayBackup, int size, int width, int r )
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

float boxfilter_gpu1( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = size/width > BLOCKDIM_MAX?BLOCKDIM_MAX: size/width;
	int countBlock = 1;

	// 第一步，扫描累加
	boxfilter_kernel_scan1<<< 1, sizeBlock >>>( d_array, size, width );
#if 1
	// 第二步，等间隔相减
	float* d_arrayBackup ;
	cudaMalloc( (void**)&d_arrayBackup, sizeof(float)*size );
	cudaMemcpy( d_arrayBackup, d_array, sizeof(float)*size, cudaMemcpyDeviceToDevice );
	boxfilter_kernel_delta1<<< 1, sizeBlock >>>( d_array, d_arrayBackup, size, width, r );
#endif
	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return array[size-1];
}

__global__ void boxfilter_kernel2( float* array, int size, int width, int r )
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

float boxfilter_gpu2( float* array, int size, int width, int r )
{
	float* d_array ;
	cudaMalloc( (void**)&d_array, sizeof(float)*size );

	cudaMemcpy( d_array, array, sizeof(float)*size, cudaMemcpyHostToDevice );

	int sizeBlock = width>BLOCKDIM_MAX?BLOCKDIM_MAX: width;
	int countBlock = size / width ;// 行数，一个block处理一行
	boxfilter_kernel2<<< countBlock, sizeBlock >>>( d_array, size, width, r );

	cudaMemcpy( array, d_array, sizeof(float)*size, cudaMemcpyDeviceToHost );

	cudaFree( d_array );

	return array[size-1];
}

void warnup_gpu( float* array, int size, int width, int r )
{
	boxfilter_gpu2( array, size, width, r );
}
