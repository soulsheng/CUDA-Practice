
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerCPP.h"
#include "timerCUDA.h"

#include <iostream>
using namespace std;

#define N (1<<21)
#define REPEAT   100


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
	
	//--CPU �汾1��ֱ���ۼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu1( a, N );
	}
	timer.stop();

	cout << "reduction_cpu �汾1��ֱ���ۼ�" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU �汾2����������----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu2( a, N );
	}
	timer.stop();

	cout << "reduction_cpu �汾2����������" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU �汾3�������ݼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu3( a, N );
	}
	timer.stop();

	cout << "reduction_cpu �汾3�������ݼ�" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--GPU �汾1����������----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu1( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾1����������" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	//--GPU �汾2�������ݼ�----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu2( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾2�������ݼ�" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	
	//--GPU �汾3��shared----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu2( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾3��shared " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	free( a );
}