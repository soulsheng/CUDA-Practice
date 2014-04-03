
#include "vectorDot_cpu.h"
#include "vectorDot_gpu.cuh"
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
	
	//--CPU �汾1��ֱ���ۼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		vectorDot_cpu1( a, b, c, N );
	}
	timer.stop();

	cout << "vectorDot_cpu �汾1��ֱ���ۼ�" << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( c, N );
#if 0
	//--CPU �汾2����������----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = vectorDot_cpu2( a, b,  N );
	}
	timer.stop();

	cout << "vectorDot_cpu �汾2����������" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU �汾3�������ݼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = vectorDot_cpu3( a, b,  N );
	}
	timer.stop();

	cout << "vectorDot_cpu �汾3�������ݼ�" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );
#endif
	//--GPU �汾1��ֱ���ۼ�----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		vectorDot_gpu1( a,  b, c, N );
	}
	timerGPU.stop();

	cout << "vectorDot_gpu �汾1��ֱ���ۼ�" << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( c, N );
#if 0
	//--GPU �汾2�������ݼ�----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = vectorDot_gpu2( a,  b, N );
	}
	timerGPU.stop();

	cout << "vectorDot_gpu �汾2�������ݼ�" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	
	//--GPU �汾3��shared----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = vectorDot_gpu2( a,  b, N );
	}
	timerGPU.stop();

	cout << "vectorDot_gpu �汾3��shared " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );
#endif
	free( a ); free( b ); free( c );
}