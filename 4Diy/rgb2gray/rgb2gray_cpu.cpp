
#include "rgb2gray_cpu.h"

#include <iostream>
using namespace std;

#include "defineMacro.h"

void rgb2gray_cpu1( float* rgb, float* gray, int size )
{
	for (int i=0;i<size;i++)
	{
		gray[i] = 
			rgb[3*i]*  R_RATIO + 
			rgb[3*i+1]* G_RATIO + 
			rgb[3*i+2]* B_RATIO ;
	}
}

void rgb2gray_cpu2( float* rgb, float* gray, int size )
{
	for (int i=0;i<size;i++)
	{
		gray[i] = 
			rgb[i]*  R_RATIO + 
			rgb[i+1*size]* G_RATIO + 
			rgb[i+2*size]* B_RATIO ;
	}
}

void setArray( float* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		array[i] = i%256;
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
