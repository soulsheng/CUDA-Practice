
#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#include "matrixmulGPU.cuh"

#define  TILE 16

__global__ void kernelMatrixMul2( float* a, float*b, float*c, int n )
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if( ii >= n*n )
		return;

	int y= ii / n ;
	int x= ii %n;

	for(int k=0;k<n;k++)
		c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

__global__ void kernelMatrixMul1( float* a, float*b, float*c, int n )
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if( x >= n || y >=n )
		return;

	for(int k=0;k<n;k++)
		c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

__global__ void kernelMatrixMul3( float* a, float*b, float*c, int n )
{
	
	int blockIdy = blockIdx.y ;

	int blockIdxx = blockIdx.x ;
	
	int threadIdy = threadIdx.y;
	
	int threadIdxx = threadIdx.x;


			for(int k=0;k<n/TILE;k++)
			{
				// 第一步，分配小块空间
				__shared__ float aBlock[TILE][TILE];
				__shared__ float bBlock[TILE][TILE];
				__shared__ float cBlockOne[TILE][TILE];

				// 第二步，获取小块在大块中的存储位置
				int aOffset =( blockIdy * n/TILE) * (TILE*TILE) + k*TILE  ;
				int bOffset =( k * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;
				int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;

				// 第三步，小块赋值
				{
					aBlock[threadIdy][threadIdxx] = a[ aOffset + threadIdy*n + threadIdxx ];
					bBlock[threadIdy][threadIdxx] = b[ bOffset + threadIdy*n + threadIdxx ];
					cBlockOne[threadIdy][threadIdxx] = 0.0f ;
				}
				__syncthreads();

				// 第四步，小矩阵相乘		
				for(int p=0;p<TILE;p++)
				{
					cBlockOne[threadIdy][threadIdxx] += aBlock[threadIdy][p] * bBlock[p][threadIdxx];
				}
			
				// 第五步，小矩阵相乘结果累加到大矩阵		
				c[ cOffset + threadIdy*n + threadIdxx] += cBlockOne[threadIdy][threadIdxx];
			}

}


__global__ void kernelMatrixMul4( float* a, float*b, float*c, int n )
{
	
	int blockIdy = blockIdx.y ;

	int blockIdxx = blockIdx.x ;
	
	int threadIdy = threadIdx.y;
	
	int threadIdxx = threadIdx.x;

	float cBlockOne = 0.0f;

			for(int k=0;k<n/TILE;k++)
			{
				// 第一步，分配小块空间
				__shared__ float aBlock[TILE][TILE];
				__shared__ float bBlock[TILE][TILE];

				// 第二步，获取小块在大块中的存储位置
				int aOffset =( blockIdy * n/TILE) * (TILE*TILE) + k*TILE  ;
				int bOffset =( k * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;

				// 第三步，小块赋值
				{
					aBlock[threadIdy][threadIdxx] = a[ aOffset + threadIdy*n + threadIdxx ];
					bBlock[threadIdy][threadIdxx] = b[ bOffset + threadIdy*n + threadIdxx ];
				}
				__syncthreads();

				// 第四步，小矩阵相乘		
				for(int p=0;p<TILE;p++)
				{
					cBlockOne += aBlock[threadIdy][p] * bBlock[p][threadIdxx];
				}
			
				
			}

	int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdxx*TILE  ;
	// 第五步，小矩阵相乘结果累加到大矩阵		
	c[ cOffset + threadIdy*n + threadIdxx] = cBlockOne;

}

void matrixMulGPU2( float* a, float*b, float*c, int n )
{
	float *aDev,*bDev,*cDev;
	cudaMalloc( (void**)&aDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&bDev, n*n*sizeof(float) );
	cudaMalloc( (void**)&cDev, n*n*sizeof(float) );

	cudaMemcpy( aDev, a, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, b, n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( cDev, 0, n*n*sizeof(float) );

	int nBlock = 256;
	int nGrid = (n*n + nBlock-1)/nBlock;

	kernelMatrixMul2<<< nGrid, nBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );

}

void matrixMulGPU1( float* a, float*b, float*c, int n )
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

	kernelMatrixMul1<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}

void matrixMulGPU3( float* a, float*b, float*c, int n )
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

	kernelMatrixMul3<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}


void matrixMulGPU4( float* a, float*b, float*c, int n )
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

	kernelMatrixMul4<<< sizeGrid,sizeBlock >>>( aDev, bDev, cDev, n );
	cudaError_t err = cudaGetLastError();

	if( err != cudaSuccess )
		cout << "error" << endl;

	cudaMemcpy( c, cDev, n*n*sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );
}