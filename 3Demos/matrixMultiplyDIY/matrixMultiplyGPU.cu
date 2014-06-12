
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
template <int BLOCK_SIZE> 
__global__ void kernelMatrixMul3( float *A, float *B, float *C, int wA, int wB )
{
	
    // Block index
	int bx = blockIdx.x ;
	int by = blockIdx.y ;
	
    // Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;


    // Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by ;

    // Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
	float Csub = 0.0f;

	for(int a = aBegin, b = bBegin; 
		a <= aEnd; 
		a += aStep, b += bStep)
	{
		// ��һ��������С��ռ�
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// ��������С�鸳ֵ
		As[ty][tx] = A[ a + wA * ty+ tx ];
		Bs[ty][tx] = B[ b + wB * ty+ tx ];
		
		__syncthreads();

		// ���Ĳ���С�������		
#pragma unroll
		for(int k = 0; k < BLOCK_SIZE; k++ )
		{
			Csub += As[ty][k] * Bs[k][tx];
		}
		
		// Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
	}

	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	// ���岽��С������˽���ۼӵ������		
	C[ c + wB * ty + tx] = Csub;

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

	kernelMatrixMul3<16><<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n, n );
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