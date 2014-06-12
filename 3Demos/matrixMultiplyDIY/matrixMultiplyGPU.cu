
#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#include "matrixMultiplyGPU.cuh"
#include "timerCUDA.h"

#define  TILE 16

// GPU �汾1����ʼ
__global__ void kernelMatrixMul1( float* a, float*b, float*c, int n )
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if( x >= n || y >=n )
		return;

	for(int k=0;k<n;k++)
		c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

// GPU �汾2��block�ֿ飬DIY
__global__ void kernelMatrixMul2( float* a, float*b, float*c, int n )
{
	
	int blockIdy = blockIdx.y ;

	int blockIdxx = blockIdx.x ;
	
	int threadIdy = threadIdx.y;
	
	int threadIdxx = threadIdx.x;

	float cBlockOne = 0.0f;

	for(int k=0;k<n/TILE;k++)
	{
		// ��һ��������С��ռ�
		__shared__ float aBlock[TILE][TILE];
		__shared__ float bBlock[TILE][TILE];

		// �ڶ�������ȡС���ڴ���еĴ洢λ��
		int aOffset =( blockIdy * n/TILE) * (TILE*TILE) + k*TILE  ;
		int bOffset =( k * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;

		// ��������С�鸳ֵ
		{
			aBlock[threadIdy][threadIdxx] = a[ aOffset + threadIdy*n + threadIdxx ];
			bBlock[threadIdy][threadIdxx] = b[ bOffset + threadIdy*n + threadIdxx ];
		}
		__syncthreads();

		// ���Ĳ���С�������		
		for(int p=0;p<TILE;p++)
		{
			cBlockOne += aBlock[threadIdy][p] * bBlock[p][threadIdxx];
		}
			
				
	}

	int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;
	// ���岽��С������˽���ۼӵ������		
	c[ cOffset + threadIdy*n + threadIdxx] = cBlockOne;

}

// GPU �汾3��block�ֿ飬SDK
__global__ void kernelMatrixMul3( float* a, float*b, float*c, int n )
{
	
	int blockIdy = blockIdx.y ;

	int blockIdxx = blockIdx.x ;
	
	int threadIdy = threadIdx.y;
	
	int threadIdxx = threadIdx.x;

	float cBlockOne = 0.0f;

	int aBegin = blockIdy * TILE * n;
	int aStep  = TILE;
	int aEnd   = aBegin + n - 1;
	int bBegin = blockIdxx* TILE;
	int bStep  = TILE * n;

	for(int aOffset=aBegin, bOffset=bBegin; aOffset<aEnd; aOffset+=aStep, bOffset+=bStep)
	{
		// ��һ��������С��ռ�
		__shared__ float aBlock[TILE][TILE];
		__shared__ float bBlock[TILE][TILE];

		// ��������С�鸳ֵ
		{
			aBlock[threadIdy][threadIdxx] = a[ aOffset + threadIdy*n + threadIdxx ];
			bBlock[threadIdy][threadIdxx] = b[ bOffset + threadIdy*n + threadIdxx ];
		}
		__syncthreads();

		// ���Ĳ���С�������		
		for(int p=0;p<TILE;p++)
		{
			cBlockOne += aBlock[threadIdy][p] * bBlock[p][threadIdxx];
		}
						
	}

	int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;
	// ���岽��С������˽���ۼӵ������		
	c[ cOffset + threadIdy*n + threadIdxx] = cBlockOne;

}

// GPU �汾1����ʼ
void matrixMulGPU1( float* a, float*b, float*c, int n, bool bTimeKernel )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 16;
	int nGrid = (n + nBlock-1)/nBlock;
	dim3 sizeBlock(nBlock, nBlock);
	dim3 sizeGrid( nGrid, nGrid );
	
	timerCUDA	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	kernelMatrixMul1<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;
	
	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}

// GPU �汾2��block�ֿ�
void matrixMulGPU2( float* a, float*b, float*c, int n, bool bTimeKernel )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 16;
	int nGrid = (n + nBlock-1)/nBlock;
	dim3 sizeBlock(nBlock, nBlock);
	dim3 sizeGrid( nGrid, nGrid );

	timerCUDA	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	kernelMatrixMul2<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}

// GPU �汾3��block�ֿ飬SDK
void matrixMulGPU3( float* a, float*b, float*c, int n, bool bTimeKernel )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 16;
	int nGrid = (n + nBlock-1)/nBlock;
	dim3 sizeBlock(nBlock, nBlock);
	dim3 sizeGrid( nGrid, nGrid );

	timerCUDA	timerGPU;
	if( bTimeKernel )
		timerGPU.start();

	kernelMatrixMul3<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	if( bTimeKernel )
	{
		timerGPU.stop();
		cout << "Kernel time : " << timerGPU.getTime() << endl;
	}

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}

void setupCUDA()
{
	cudaSetDevice( 0 );
}