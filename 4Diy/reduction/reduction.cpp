
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerCPP.h"
#include "timerCUDA.h"

#include <iostream>
using namespace std;

#define N (1<<8)
#define REPEAT   100


void main()
{
	timerC  timer;
	timerC  timerGPU;
	
	unsigned int *a = (unsigned int *)malloc( N*sizeof(unsigned int) );
	setArray( a, N );

	warnup_gpu( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	unsigned int result = 0;
	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_cpu( a, N );
	}
	timer.stop();

	cout << "reduction_cpu: " << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	timerGPU.start();
	for(int i=0;i<REPEAT;i++)
	{
		setArray( a, N );
		result = reduction_gpu( a, N );
	}
	timerGPU.stop();

	cout << "reduction_gpu: " << result << ", timer: " << timerGPU.getTime()/REPEAT <<endl;
	printArray( a, N );

	free( a );
}