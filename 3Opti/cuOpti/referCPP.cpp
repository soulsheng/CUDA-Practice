
#include "referCPP.h"

void matrixMul1( float* a, float*b, float*c, int n )
{
	for(int y=0; y<n; y++)
		for(int x=0;x<n;x++)
			for(int k=0;k<n;k++)
				c[ y*n +x] += a[ y*n + k] * b[ k*n +x] ;
}

void matrixMul2( float* a, float*b, float*c, int n )
{
	for(int ii=0; ii<n*n; ii++)
	{
		int i= ii / n ;
		int j= ii %n;

			for(int k=0;k<n;k++)
				c[ i*n +j] += a[ i*n + k] * b[ k*n +j] ;
	}
}

#define  TILE 16
void matrixMul3( float* a, float*b, float*c, int n )
{

	for(int blockIdy=0; blockIdy<n/TILE; blockIdy++)
	{
		for(int blockIdx=0;blockIdx<n/TILE;blockIdx++)
		{
			for(int k=0;k<n/TILE;k++)
			{
				float aBlock[TILE][TILE];
				float bBlock[TILE][TILE];
				float cBlockOne[TILE][TILE];
			
				// 
				int aOffset =( blockIdy * n/TILE) * (TILE*TILE) + k*TILE  ;
				int bOffset =( k * n/TILE) * (TILE*TILE) + blockIdx*TILE  ;
				int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdx*TILE  ;

				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						{
							aBlock[threadIdy][threadIdx] = a[ aOffset + threadIdy*n + threadIdx ];
							bBlock[threadIdy][threadIdx] = b[ bOffset + threadIdy*n + threadIdx ];
							cBlockOne[threadIdy][threadIdx] = 0.0f ;
						}


				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						for(int p=0;p<TILE;p++)
					{
						cBlockOne[threadIdy][threadIdx] += aBlock[threadIdy][p] * bBlock[p][threadIdx];
					}
			
				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						c[ cOffset + threadIdy*n + threadIdx] += cBlockOne[threadIdy][threadIdx];
			}
		}
	}
}
