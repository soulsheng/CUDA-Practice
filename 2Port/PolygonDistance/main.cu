
#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include "helper_timer.h"

// Step 3: 函数声明
void polygonDistance( float x0, float y0, float *x, float *y, int n, float *d );
void polygonDistance_ref( float x0, float y0, float *x, float *y, int n, float *d );
bool verify( float *d, float *d_ref, int n );


void main()
{
	cout << "Polygon Distance begin: " << endl;

	// C编程模型：1.变量定义、2.数组申请、3.函数声明、4.函数实现、5.函数调用

	cout << "Step 1: 变量定义" << endl;
	
	// 一点 p0(x0,y0)
	float x0 = 1.0f;
	float y0 = 2.0f;

	int n = 16*1024*1024;

	cout << "Step 2: 数组申请" << endl;

	// 多边形顶点 pi(xi,yi)
	float *x = (float*)malloc( n * sizeof(float) );
	float *y = (float*)malloc( n * sizeof(float) );
	float *d = (float*)malloc( n * sizeof(float) );
	float *d_ref = (float*)malloc( n * sizeof(float) );

	for (int i=0;i<n;i++)
	{
		x[i] = sin( (float)i/n );
		y[i] = cos( (float)i/n );
		d[i] = 0.0f;
		d_ref[i] = 0.0f;
	}

	cout << "Step 5: 函数调用" << endl;

	StopWatchInterface *watchTime;
	sdkCreateTimer( &watchTime );

	// test time of gpu
	sdkStartTimer( &watchTime );
	
	polygonDistance( x0, y0, x, y, n, d);
	cudaDeviceSynchronize();

	sdkStopTimer( &watchTime );
	cout << sdkGetTimerValue( &watchTime ) << " ms" << endl;

	// test time of cpu
	sdkResetTimer( &watchTime );
	sdkStartTimer( &watchTime );

	polygonDistance_ref( x0, y0, x, y, n, d_ref);

	sdkStopTimer( &watchTime );
	cout << sdkGetTimerValue( &watchTime ) << " ms" << endl;


	if( verify( d, d_ref, n ) )
		cout << "verify passed!" << endl;
	else
		cout << "verify failed!" << endl;

	cout << "Polygon Distance end! " << endl;
}

__constant__ float c_x0[1], c_y0[1];

__global__
void polygonDistance_kernel( float *x, float *y, int n, float *d )
{
	int i = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	if( i >= n ) return;
	d[i] = sqrt( (x[i]-*c_x0)*(x[i]-*c_x0) + (y[i]-*c_y0)*(y[i]-*c_y0) );
}

void polygonDistance_body( float x0, float y0, float *x, float *y, int n, float *d )
{
	//for ( int i=0; i<n; i++ )
	dim3 block( 1024 );
	dim3 grid( (n+1024*32-1)/1024/32, 32 );
	{
		polygonDistance_kernel<<<grid,block>>>( x, y, n, d );
	}
}

// Step 4: 函数实现
void polygonDistance( float x0, float y0, float *x, float *y, int n, float *d )
{
	cout << "Step 3: 函数声明" << endl;
	cout << "Step 4: 函数实现" << endl;

	float *gpu_x, *gpu_y, *gpu_d;
	cudaMalloc( &gpu_x, n * sizeof(float) );
	cudaMalloc( &gpu_y, n * sizeof(float) );
	cudaMalloc( &gpu_d, n * sizeof(float) );

	cudaMemcpy( gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_y, y, n * sizeof(float), cudaMemcpyHostToDevice );

	cudaMemcpyToSymbol( c_x0, &x0, 1 * sizeof(float) );
	cudaMemcpyToSymbol( c_y0, &y0, 1 * sizeof(float) );

	polygonDistance_body( x0, y0, gpu_x, gpu_y, n, gpu_d);
	
	cudaMemcpy( d, gpu_d, n * sizeof(float), cudaMemcpyDeviceToHost );

	cudaError err = cudaGetLastError();
	if( err != cudaSuccess )
		cout << "error" << endl;
}

void polygonDistance_ref( float x0, float y0, float *x, float *y, int n, float *d_ref )
{
	for ( int i=0; i<n; i++ )
	{
		d_ref[i] = sqrt( (x[i]-x0)*(x[i]-x0) + (y[i]-y0)*(y[i]-y0) );
	}	
}

bool verify( float *d, float *d_ref, int n )
{
	for ( int i=0; i<n; i++ )
	{
		if( fabs( d[i] - d_ref[i] ) > 1e-5 )
			return false;
	}

	return true;
}
