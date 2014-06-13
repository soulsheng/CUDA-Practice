#include "matrixTranspose.h"



__global__ void transposeSimple(float *odata, float *idata, int size_x, int size_y)
{
	//2D thread ID	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int index_in = row*size_x + col;
	int index_out = col*size_x + row;

	odata[index_out] =  idata[index_in];
}

__global__ void transposeSharedBlock(float *odata, float *idata, int size_x, int size_y)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1]; // +1 ½â¾öbank confict

    int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int index_in = xIndex + (yIndex)*size_x;

    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int index_out = xIndex + (yIndex)*size_y;

    for (int i=0; i<BLOCK_SIZE; i+=BLOCK_SIZE)
    {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*size_x];
    }
    
    __syncthreads();

    for (int i=0; i<BLOCK_SIZE; i+=BLOCK_SIZE)
    {
      odata[index_out+i*size_y] = tile[threadIdx.x][threadIdx.y+i];
    }
}

extern "C"
void MatrixTransposeOnDevice(float *P, float *M, unsigned int size_x, unsigned int size_y, bool bTimerKernel/*=false*/)
{
	int size= size_x*size_y*sizeof(float);

	//Interface host call to the device kernel code and invoke the kernel

	float *M_d=NULL;
    cudaMalloc((void**)&M_d,size);
	cudaMemcpy(M_d, M,size,cudaMemcpyHostToDevice);

	float *P_d=NULL;
	cudaMalloc((void**)&P_d,size);
	cudaMemcpy(P_d, P,size,cudaMemcpyHostToDevice);

	//kernel invocation code
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid(size_x/BLOCK_SIZE,size_y/BLOCK_SIZE);


	cudaEvent_t start, stop;

	if( bTimerKernel )
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Record the start event
		cudaEventRecord(start, NULL);
	}

#if 0
	//Matrix Transpose Kernel Simple
	transposeSimple<<<dimGrid,dimBlock>>>(P_d,M_d,size_x,size_y);
#else
	//Matrix Transpose Kernel Shared
	transposeSharedBlock<<<dimGrid,dimBlock>>>(P_d,M_d,size_x,size_y);
#endif

	if( bTimerKernel )
	{
		// Record the stop event
		cudaEventRecord(stop, NULL);

		// Wait for the stop event to complete
		cudaEventSynchronize(stop);

		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);

		printf(	"GPU Time Kernel = %.3f msec \n",msecTotal );
	}

	//read P from the device
	cudaMemcpy(P,P_d,size,cudaMemcpyDeviceToHost);

	//free device matrices
	cudaFree(M_d);
	cudaFree(P_d);

}