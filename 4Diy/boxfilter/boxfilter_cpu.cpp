
#include "boxfilter_cpu.h"

#include <iostream>
using namespace std;

#define  PRINTMAX   1024

float boxfilter_cpu2( float* array, int size, int width )
{
	float* temp = (float*)malloc( size* sizeof(float) );
	temp[0] = array[0];
	float* pSrc = NULL;
	float* pDst = NULL;

	float result = 0;
	for ( int d=1, round=0;d<=size/2; d+=d, round++ )
	{
		if ( round%2 )
		{
			pSrc = temp;
			pDst = array;
		}
		else
		{
			pSrc = array;
			pDst = temp;
		}

		for ( int i=d;i< size; i++ )
		{
			pDst[i] = pSrc[i] + pSrc[i-d];
		}

		for( int i=0;i<d;i++)
			pDst[i] = pSrc[i];
	}
	
	if (pDst!=array)
	{
		memcpy( array, pDst, size * sizeof(float) );
	}

	result = pDst[size-1];
	free(temp);
	return result;
}

float boxfilter_cpu1( float* array, int size, int width )
{
	float result = 0;
	for (int row=0;row<size/width;row++)
		for (int i=1;i<width;i++)
		{
			array[i+row*width] += array[i-1+row*width];
		}

	return array[size-1];
}

void setArray( float* array, int size )
{
	for ( int i=0;i<size; i++)
	{
		array[i] = 1;
	}
}

void printArray( float* array, int size, int width )
{
	if ( size>PRINTMAX )
	{
		return;
	}
	for ( int i=0;i<size; i++)
	{
		cout << array[i] << " ";
		if ( (i+1)%width==0)
		{
			cout << endl;
		}
	}
	cout << endl << endl;
}
