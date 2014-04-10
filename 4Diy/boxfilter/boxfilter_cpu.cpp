
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

float boxfilter_cpu1( float* array, int size, int width, int r )
{
	float result = 0;

	// 第一步，扫描累加
	for (int row=0;row<size/width;row++)
	{
		for (int i=1;i<width;i++)
		{
			array[i+row*width] += array[i-1+row*width];
		}
	}
#if 1	
	float* arrayTemp = (float*)malloc( size*sizeof(float) );
	memcpy(arrayTemp, array, size*sizeof(float) );

	// 第二步，等间隔相减
	int nRight=0, nLeft=0;
	for (int row=0;row<size/width;row++)
	{
		for (int i=0;i<width;i++)
		{
			if(i<=r)
			{
				nLeft = 0;
				nRight = i+r;
				array[i+row*width] = arrayTemp[row*width + nRight] ;
			}
			else if( i>r && i<width-r )
			{
				nLeft = i-r-1;
				nRight = i+r;
				array[i+row*width] = arrayTemp[row*width + nRight] - arrayTemp[row*width + nLeft];
			}
			else//if( i>width-r && i<width )
			{
				nLeft = i-r-1;
				nRight = width-1;
				array[i+row*width] = arrayTemp[row*width + nRight] - arrayTemp[row*width + nLeft];
			}
		}
	}

	free(arrayTemp);
#endif
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
