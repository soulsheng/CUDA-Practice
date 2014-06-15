
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerTest.h"

#include <iostream>
using namespace std;

#define N (1<<24)
#define REPEAT   10


void main()
{
	timerTest  timer;
	timerTestCU  timerGPU;
	
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

	//--CPU �汾2���粽���ۼӣ���������----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu3( a, N );
	}
	timer.stop();

	cout << "reduction_cpu �汾2���粽���ۼӣ���������" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	//--CPU �汾3���粽���ۼӣ������ݼ�----------	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu3( a, N );
	}
	timer.stop();

	cout << "reduction_cpu �汾3���粽���ۼӣ������ݼ�" << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	cout << endl << endl;

	//--GPU �汾1���粽���ۼӣ���������----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu1( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾1���粽���ۼӣ���������" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	// kernel time - reduction_gpu �汾1
	reduction_gpu1( a, N, true );

	//--GPU �汾2���粽���ۼӣ������ݼ�----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu2( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾2���粽���ۼӣ������ݼ�" << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	// kernel time - reduction_gpu �汾2
	reduction_gpu2( a, N, true );

	
	//--GPU �汾3��shared memory----------	
	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu3( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu �汾3��shared memory " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	// kernel time - reduction_gpu �汾3
	reduction_gpu3( a, N, true );

	free( a );
}