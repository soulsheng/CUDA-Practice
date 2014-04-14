
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
	
	//--CPU �汾1��ֱ���ۼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_cpu1( a, N, WIDTH, RADIUS );
	}
	timer.stop();

	cout << "boxfilter_cpu1 �汾1��ֱ���ۼ�" << ", timer: " << timer.getTime()/REPEAT << endl << endl;
	printArray( a, N, WIDTH );

	//--CPU �汾2��scan�����----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_cpu2( a, N, WIDTH, RADIUS );
	}
	timer.stop();

	cout << "boxfilter_cpu �汾2��scan�����" << ", timer: " << timer.getTime()/REPEAT << endl << endl;
	printArray( a, N, WIDTH );

	setArray( a, N );
	warnup_gpu( a, N, WIDTH );

	//--GPU �汾1s��ֱ���ۼ� �д���----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu1s( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾1s��ֱ���ۼ� �д��� " <<endl
		<< "kernel timer: " << result  <<endl
		<< "Memory timer: " << timerGPU.getTime()/REPEAT - result  <<endl
		<< "GPU timer: " 	<< timerGPU.getTime()/REPEAT <<endl<<endl;
	printArray( a, N, WIDTH );

	//--GPU �汾1p��ֱ���ۼ� �в���----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu1p( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾1p��ֱ���ۼ� �в��� " <<endl
		<< "kernel timer: " << result  <<endl
		<< "Memory timer: " << timerGPU.getTime()/REPEAT - result  <<endl
		<< "GPU timer: " 	<< timerGPU.getTime()/REPEAT <<endl<<endl;
	printArray( a, N, WIDTH );

	//--GPU �汾2s��scan����� �д���----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu2s( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾2s��scan����� �д��� " <<endl
		<< "kernel timer: " << result  <<endl
		<< "Memory timer: " << timerGPU.getTime()/REPEAT - result  <<endl
		<< "GPU timer: " 	<< timerGPU.getTime()/REPEAT <<endl<<endl;
	printArray( a, N, WIDTH );

	//--GPU �汾2p��scan����� �в���----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = boxfilter_gpu2p( a, N, WIDTH, RADIUS );
	}
	timerGPU.stop();

	cout << "boxfilter_gpu �汾2p��scan����� �в��� " <<endl
		<< "kernel timer: " << result  <<endl
		<< "Memory timer: " << timerGPU.getTime()/REPEAT - result  <<endl
		<< "GPU timer: " 	<< timerGPU.getTime()/REPEAT <<endl<<endl;
	printArray( a, N, WIDTH );

	free( a );
}