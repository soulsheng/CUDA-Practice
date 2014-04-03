
#include "reduction_cpu.h"

#include <iostream>
using namespace std;

unsigned int reduction_cpu2( unsigned int* array, int size )
{
	unsigned int result = 0;
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

unsigned int reduction_cpu( unsigned int* array, int size )
{
	unsigned int result = 0;
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

unsigned int reduction_cpu1( unsigned int* array, int size )
{
	unsigned int result = 0;
	for (int i=0;i<size;i++)
	{
		result += array[i];
	}

	return result;
}

void setArray( unsigned int* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		array[i] = 1;
	}
}

void printArray( unsigned int* array, int size )
{
	if ( size>256 )
	{
		return;
	}
	for ( int i=0;i<size; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl << endl;
}
