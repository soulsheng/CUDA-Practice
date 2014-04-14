
#include "boxfilter_cpu.h"

#include <iostream>
using namespace std;

#define  PRINTMAX   1024

float boxfilter_cpu1( float* array, int size, int width, int r )
{
	float* temp = (float*)malloc( size* sizeof(float) );
	memcpy( temp, array, size * sizeof(float) );
	memset( array, 0, size * sizeof(float) );

	for (int row=0;row<size/width;row++)
	{
		for (int i=0;i<width;i++)
		{
			if(i-r>=0 && i+r<=width-1)
			{
				for(int j=i-r;j<=i+r;j++)
					array[i+row*width] += temp[row*width+j];
			}
			else if(i-r<0)
			{
				for(int j=0;j<=i+r;j++)
					array[i+row*width] += temp[row*width+j];
			}
			else
			{
				for(int j=i-r;j<=width-1;j++)
					array[i+row*width] += temp[row*width+j];
			}
		}
	}

	free(temp);
	return array[size-1];
}

float boxfilter_cpu2( float* array, int size, int width, int r )
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
	float* arrayBackup = (float*)malloc( size*sizeof(float) );
	memcpy(arrayBackup, array, size*sizeof(float) );

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
				array[i+row*width] = arrayBackup[row*width + nRight] ;
			}
			else if( i>r && i<width-r )
			{
				nLeft = i-r-1;
				nRight = i+r;
				array[i+row*width] = arrayBackup[row*width + nRight] - arrayBackup[row*width + nLeft];
			}
			else//if( i>width-r && i<width )
			{
				nLeft = i-r-1;
				nRight = width-1;
				array[i+row*width] = arrayBackup[row*width + nRight] - arrayBackup[row*width + nLeft];
			}
		}
	}

	free(arrayBackup);
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
