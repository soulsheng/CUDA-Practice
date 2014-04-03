
#include "reduction_cpu.h"
#include "reduction_gpu.cuh"

#include <iostream>
using namespace std;

#define N 16



void main()
{
	int *a = (int *)malloc( N*sizeof(int) );
	setArray( a, N );

	cout << "size: " << N << endl << endl ;

	cout << "setArray" <<endl;
	printArray( a, N );

	int result = 0;
	
	result = reduction_cpu( a, N );

	cout << "reduction_cpu" <<endl;
	printArray( a, N );

	result = reduction_gpu( a, N );

	cout << "reduction_gpu" <<endl;
	printArray( a, N );


}