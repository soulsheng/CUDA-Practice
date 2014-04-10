
#include "rgb2gray_cpu.h"

#include <iostream>
using namespace std;

#define  PRINTMAX   1024

void rgb2gray_cpu1( float* arrayA, float* arrayB, float* arrayC, int size )
{
	for (int i=0;i<size;i++)
	{
		arrayC[i] = 
			arrayA[3*i]*arrayB[3*i] + 
			arrayA[3*i+1]*arrayB[3*i+1] + 
			arrayA[3*i+1]*arrayB[3*i+1];
	}
}

void setArray( float* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		array[i] = 1;
	}
}

void printArray( float* array, int size )
{
	if ( size>PRINTMAX )
	{
		return;
	}
	for ( int i=0;i<size; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl << endl;
}
