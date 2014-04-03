
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#define N (1<<24)
#define REPEAT   100


void main()
{
	timerC  timer;

	int *a = (int *)malloc( N*sizeof(int) );
	setArray( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	unsigned int result = 0;
	
	timer.start();
	for(int i=0;i<REPEAT;i++)
	result = reduction_cpu( a, N );
	timer.stop();

	cout << "reduction_cpu: " << result << ", timer: " << timer.getTime()/REPEAT << endl;
	printArray( a, N );

	result = reduction_gpu( a, N );

	cout << "reduction_gpu: " << result <<endl;
	printArray( a, N );

	free( a );
}