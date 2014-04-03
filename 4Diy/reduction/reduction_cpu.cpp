
#include "reduction_cpu.h"

#include <iostream>
using namespace std;

int reduction_cpu( int* array, int size )
{
	int result = 0;
	for (int i=0;i<size;i++)
	{
		result += array[i];
	}

	return result;
}

void setArray( int* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		array[i] = 1;
	}
}

void printArray( int* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl << endl;
}
