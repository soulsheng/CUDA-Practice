
#include "rgb2gray_cpu.h"
#include "rgb2gray_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#define N (1<<21)
#define REPEAT   10


void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	float *a = (float *)malloc( N*sizeof(float)*3 );
	setArray( a, N*3 );

	float *b = (float *)malloc( N*sizeof(float)*3 );
	setArray( b, N*3 );

	float *c = (float *)malloc( N*sizeof(float) );
	setArray( c, N );

	warnup_gpu( a, b, c, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	float result = 0;
	
	//--CPU 版本1：直接累加----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		rgb2gray_cpu1( a, b, c, N );
	}
	timer.stop();

	cout << "rgb2gray_cpu 版本1：直接累加" << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( c, N );
#if 0
	//--CPU 版本2：步长递增----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = rgb2gray_cpu2( a, b,  N );
	}
	timer.stop();

	cout << "rgb2gray_cpu 版本2：步长递增" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU 版本3：步长递减----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = rgb2gray_cpu3( a, b,  N );
	}
	timer.stop();

	cout << "rgb2gray_cpu 版本3：步长递减" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );
#endif
	//--GPU 版本1：直接累加----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		rgb2gray_gpu1( a,  b, c, N );
	}
	timerGPU.stop();

	cout << "rgb2gray_gpu 版本1：直接累加" << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( c, N );
	
	//--GPU 版本2：合并----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		rgb2gray_gpu2( a,  b, c, N );
	}
	timerGPU.stop();

	cout << "rgb2gray_gpu 版本2：合并" << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

#if 0
	
	//--GPU 版本3：shared----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = rgb2gray_gpu2( a,  b, N );
	}
	timerGPU.stop();

	cout << "rgb2gray_gpu 版本3：shared " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );
#endif
	free( a ); free( b ); free( c );
}