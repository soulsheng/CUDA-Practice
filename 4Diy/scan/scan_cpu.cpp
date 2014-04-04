
#include "scan_cpu.h"

#include <iostream>
using namespace std;

#define  PRINTMAX   1024
float scan_cpu3( float* array, int size )
{
	float result = 0;
	for ( int d=size/2;d>=1; d=d/2 )
	{
		for ( int i=0;i< d; i++ )
		{
			array[i] += array[i+d];
		}
	}

	result = array[0];
	return result;
}

float scan_cpu2( float* array, int size )
{
	float result = 0;
	for ( int d=1;d<=size/2; d=d*2 )
	{
		for ( int i=0;i< size; i+=2*d )
		{
			array[i] += array[i+d];
		}
	}

	result = array[0];
	return result;
}

float scan_cpu1( float* array, int size )
{
	float result = 0;
	for (int i=0;i<size;i++)
	{
		result += array[i];
	}

	return result;
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
