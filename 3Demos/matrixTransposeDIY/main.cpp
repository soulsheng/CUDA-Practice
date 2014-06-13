#include "matrixTranspose.h"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward
extern "C" void MatrixTransposeOnDevice(float*,float*,unsigned int,unsigned int, bool bTimerKernel=false);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void computeTransposeGold(float *gold, float *idata,const  int size_x, const  int size_y)
{
	for (int y = 0; y < size_y; ++y)
	{
		for (int x = 0; x < size_x; ++x)
		{
			gold[(x * size_y) + y] = idata[(y * size_x) + x];
		}
	}
}

void Prinf(float *t)
{
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			int temp=i*width+j;
			printf("t[%d][%d]=%f   ",i,j,t[temp]);
		}
		printf("\n");
	}
}

int main(){

	//Cpu Cost Time Var
	DWORD CpuStartTime,CpuStopTime;

	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = width*height;
	cudaError_t error;

	// Matrices M for the program
	// Allocate and initialize the matrices
	float *M=NULL;
	M = (float*) malloc(size_elements*sizeof(float));
	if (M)
	{
		for(unsigned int i = 0; i < size_elements; i++) 
		    M[i] = 1.0f * i;
	} 

	// Matrices P for the result
	float *P=NULL;
	P = (float*) malloc(size_elements*sizeof(float));
	if (P)
	{
		for(unsigned int i = 0; i < size_elements; i++) 
			P[i] = 0.0f;
	}

	int size= width*height*sizeof(float);
	printf("(width:%d, height:%d), data size=%.3f MBytes\n",width, height, size/(1000000.0f));

	// warmup
	MatrixTransposeOnDevice(P,M,width,height);

	// Start Timing
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start,stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Record the start event
	error = cudaEventRecord(start, NULL);
	

	for (int i=0; i < NUM_REPS; i++)
	{
		// transpose on the device
		MatrixTransposeOnDevice(P,M,width,height);
	}

	// Record the stop event
	cudaEventRecord(stop, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal;
	printf(	"\nGPU Time Total = %.3f msec \n",msecPerMatrixMul/NUM_REPS);

	// Test Kernel Time 
	MatrixTransposeOnDevice(P,M,width,height, true );

	//compute the matrix multiplication on the CPU for comparison
	float *reference = NULL;
	reference = (float*) malloc(size_elements*sizeof(float));
	if (reference)
	{
		for(unsigned int i = 0; i < size_elements; i++) 
			reference[i] = 0.0f;
	}
	//Prinf(reference);
	// Start Timing of CPU

	CpuStartTime = timeGetTime();
	computeTransposeGold(reference,M,width,height);
	CpuStopTime = timeGetTime();
	printf("\nCPU Time: %ld msec.\n",CpuStopTime-CpuStartTime);


	// test relative error by the formula 
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps 
	bool correct = true;
	double eps = 1.e-3 ; // machine zero
	for (int i = 0; i < width* height; i++)
	{
		double abs_err = fabs(reference[i] - P[i]);
		if (abs_err > eps)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, P[i], reference[i], eps);
			correct = false;
		}
	}

	printf("\n%s\n", correct ? "Result = PASS" : "Result = FAIL");

	// output result if output file is requested

	// Free host matrices
	free(M);
	M = NULL;
	free(P);
	P = NULL;
	free(reference);
	reference=NULL;

}


