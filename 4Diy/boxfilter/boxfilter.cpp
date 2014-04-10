
#include "boxfilter_cpu.h"
#include "boxfilter_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#define WIDTH	(1<<10)
#define HEIGHT	(1<<10)

#define N (WIDTH*HEIGHT)
#define REPEAT		10
#define RADIUS		20

void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	float *a = (float *)malloc( N*sizeof(float) );
	setArray( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N, WIDTH );

	float result = 0;
	
	//--CPU �汾��ֱ���ۼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_cpu1( a, N, WIDTH, RADIUS );
	}
	timer.stop();

	cout << "boxfilter_cpu �汾��ֱ���ۼ�" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N, WIDTH );

	setArray( a, N );
	warnup_gpu( a, N, WIDTH );

	//--GPU �汾1���д���----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu1( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾1���д��� " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N, WIDTH );

	//--GPU �汾2��shared----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu2( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾2��shared " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N, WIDTH );

	free( a );
}