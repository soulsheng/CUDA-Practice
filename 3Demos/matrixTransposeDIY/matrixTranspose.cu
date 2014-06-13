#include "matrixTranspose.h"



__global__ void MatrixTransposeKernel(float *odata, float *idata, int size_x, int size_y)
{
	//2D thread ID
	int bx=blockIdx.x;
	int by= blockIdx.y;
	int tx= threadIdx.x;
	int ty= threadIdx.y;	
	
	int row=by*blockDim.y+ty;
	int col=bx*blockDim.x+tx;

	int index_in = row*size_x + col;
	int index_out = col*size_x + row;

	odata[index_out] =  idata[index_in];
}

//__global__ void MatrixTransposeKernel(float *odata, float *idata, int size_x, int size_y)
//{
//	__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
//
//	int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//	int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//	int index_in = xIndex + (yIndex)*size_x;
//
//	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
//	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
//	int index_out = xIndex + (yIndex)*size_y;
//
//	tile[threadIdx.y][threadIdx.x] = idata[index_in];
//
//	__syncthreads();
//
//	odata[index_out] = tile[threadIdx.x][threadIdx.y];
//}

extern "C"
void MatrixTransposeOnDevice(float *P, float *M, unsigned int size_x, unsigned int size_y)
{
	int size= size_x*size_y*sizeof(float);

	//Interface host call to the device kernel code and invoke the kernel

	printf("size=%d\n",size);

	float *M_d=NULL;
    cudaMalloc((void**)&M_d,size);
	cudaMemcpy(M_d, M,size,cudaMemcpyHostToDevice);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	float *P_d=NULL;
	cudaMalloc((void**)&P_d,size);
	cudaMemcpy(P_d, P,size,cudaMemcpyHostToDevice);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//kernel invocation code
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid(size_x/BLOCK_SIZE,size_y/BLOCK_SIZE);

	//Matrix Transpose Kernel
	MatrixTransposeKernel<<<dimGrid,dimBlock>>>(P_d,M_d,size_x,size_y);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//read P from the device
	cudaMemcpy(P,P_d,size,cudaMemcpyDeviceToHost);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//free device matrices
	cudaFree(M_d);
	cudaFree(P_d);

}