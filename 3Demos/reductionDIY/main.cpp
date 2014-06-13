
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerCPP.h"
#include "timerCUDA.h"

#include <iostream>
using namespace std;

#define N (1<<21)
#define REPEAT   10


void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	float *a = (float *)malloc( N*sizeof(float) );
	setArray( a, N );

	warnup_gpu( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	float result = 0;
	
	//--CPU 版本1：直接累加----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu1( a, N );
	}
	timer.stop();

	cout << "reduction_cpu 版本1：直接累加" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU 版本2：跨步长累加，步长递增----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu3( a, N );
	}
	timer.stop();

	cout << "reduction_cpu 版本2：跨步长累加，步长递增" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU 版本3：跨步长累加，步长递减----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu3( a, N );
	}
	timer.stop();

	cout << "reduction_cpu 版本3：跨步长累加，步长递减" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	cout << endl << endl;

	//--GPU 版本1：跨步长累加，步长递增----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu1( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu 版本1：跨步长累加，步长递增" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	//--GPU 版本2：跨步长累加，步长递减----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu2( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu 版本2：跨步长累加，步长递减" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	
	//--GPU 版本3：shared memory----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu3( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu 版本3：shared memory " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	free( a );
}