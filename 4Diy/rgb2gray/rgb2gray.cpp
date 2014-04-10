
#include "rgb2gray_cpu.h"
#include "rgb2gray_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#include "defineMacro.h"


void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	float *rgb = (float *)malloc( N*sizeof(float)*3 );
	setArray( rgb, N*3 );

	float *gray = (float *)malloc( N*sizeof(float) );
	setArray( gray, N );

	warnup_gpu( rgb, gray, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( rgb, N*3 );

	float result = 0;
	
	//--CPU 版本1：aos(array of struct)----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( rgb, N*3 );
		rgb2gray_cpu1( rgb, gray, N );
	}
	timer.stop();

	cout << "rgb2gray_cpu 版本1：aos(array of struct)" << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( gray, N );

	//--CPU 版本2：soa(struct of array)----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( rgb, N*3 );
		rgb2gray_cpu2( rgb, gray,  N );
	}
	timer.stop();

	cout << "rgb2gray_cpu 版本2：soa(struct of array)" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( gray, N );

	//--GPU 版本1：aos(array of struct)----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( rgb, N*3 );
		rgb2gray_gpu1( rgb, gray, N );
	}
	timerGPU.stop();

	cout << "rgb2gray_gpu 版本1：aos(array of struct)" << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( gray, N );
	
	//--GPU 版本2：soa(struct of array)----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( rgb, N*3 );
		rgb2gray_gpu2( rgb, gray, N );
	}
	timerGPU.stop();

	cout << "rgb2gray_gpu 版本2：soa(struct of array)" << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( gray, N );

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
	free( rgb ); free( gray );
}