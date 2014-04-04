
#include "scan_cpu.h"
#include "scan_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#define N (1<<20)
#define REPEAT   10


void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	float *a = (float *)malloc( N*sizeof(float) );
	setArray( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	float result = 0;
	
	//--CPU 版本：直接累加----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = scan_cpu1( a, N );
	}
	timer.stop();

	cout << "scan_cpu 版本：直接累加" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	setArray( a, N );
	warnup_gpu( a, N );

	//--GPU 版本1：行串行----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = scan_gpu1( a, N );
	}
	timerGPU.stop();

	cout << "scan_gpu 版本1：行串行 " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	//--GPU 版本2：shared----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = scan_gpu2( a, N );
	}
	timerGPU.stop();

	cout << "scan_gpu 版本2：shared " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	free( a );
}