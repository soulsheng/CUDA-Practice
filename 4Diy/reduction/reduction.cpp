
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"
#include "timerCPP.h"

#include <iostream>
using namespace std;

#define N 1024



void main()
{
	timerC  timer;

	int *a = (int *)malloc( N*sizeof(int) );
	setArray( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	int result = 0;
	
	timer.start();
	result = reduction_cpu( a, N );
	timer.stop();

	cout << "reduction_cpu: " << result << ", timer: " << timer.getTime() << endl;
	printArray( a, N );

	result = reduction_gpu( a, N );

	cout << "reduction_gpu: " << result <<endl;
	printArray( a, N );


}