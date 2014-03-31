
#include "referCPP.h"

void matrixMul1( float* a, float*b, float*c, int n )
{
	for(int i=0; i<n; i++)
		for(int j=0;j<n;j++)
			for(int k=0;k<n;k++)
				c[ i*n +j] += a[ i*n + k] * b[ k*n +j] ;
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

	for(int i=0; i<n/TILE; i++)
		for(int j=0;j<n/TILE;j++)
			for(int k=0;k<n/TILE;k++)
		{
			float aBlock[TILE][TILE];
			float bBlock[TILE][TILE];
			float cBlockOne[TILE][TILE];
			
			// 
			int aOffset =( i * n/TILE) * (TILE*TILE) + k*TILE  ;
			int bOffset =( k * n/TILE) * (TILE*TILE) + j*TILE  ;
			int cOffset =( i * n/TILE) * (TILE*TILE) + j*TILE  ;

			for(int l=0;l<TILE;l++)
				for(int m=0;m<TILE;m++)
					{
						aBlock[l][m] = a[ aOffset + l*n + m ];
						bBlock[l][m] = b[ bOffset + l*n +m ];
						cBlockOne[l][m] = 0.0f ;
					}


			for(int l=0; l<TILE;l++)
				for(int m=0;m<TILE;m++)
					for(int p=0;p<TILE;p++)
				{
					cBlockOne[l][m] += aBlock[l][p] * bBlock[p][m];
				}
			
			for(int l=0; l<TILE;l++)
				for(int m=0;m<TILE;m++)
					c[ cOffset + l*n + m] += cBlockOne[l][m];
		}
}
