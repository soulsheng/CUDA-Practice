
#include "referCPP.h"

void matrixMul1( float* a, float*b, float*c, int n )
{
	for(int y=0; y<n; y++)
		for(int x=0;x<n;x++)
		{
			for(int k=0;k<n;k++)
			{
				// ��һ��������С��ռ�
				float aBlock;
				float bBlock;
				float cBlockOne;

				// �ڶ�������ȡС���ڴ���еĴ洢λ��
				int aOffset =( y * n ) + k ;
				int bOffset =( k * n ) + x ;
				int cOffset =( y * n ) + x ;

				// ��������С�鸳ֵ
				aBlock = a[ aOffset ];
				bBlock = b[ bOffset ];
				cBlockOne = 0.0f;

				// ���Ĳ���С�������
				cBlockOne = aBlock * bBlock;

				// ���岽��С������˽���ۼӵ������
				c[ y*n +x] += cBlockOne ;
			}
		}
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
				// ��һ��������С��ռ�
				float aBlock[TILE][TILE];
				float bBlock[TILE][TILE];
				float cBlockOne[TILE][TILE];

				// �ڶ�������ȡС���ڴ���еĴ洢λ��
				int aOffset =( blockIdy * n/TILE) * (TILE*TILE) + k*TILE  ;
				int bOffset =( k * n/TILE) * (TILE*TILE) + blockIdx*TILE  ;
				int cOffset =( blockIdy * n/TILE) * (TILE*TILE) + blockIdx*TILE  ;

				// ��������С�鸳ֵ
				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						{
							aBlock[threadIdy][threadIdx] = a[ aOffset + threadIdy*n + threadIdx ];
							bBlock[threadIdy][threadIdx] = b[ bOffset + threadIdy*n + threadIdx ];
							cBlockOne[threadIdy][threadIdx] = 0.0f ;
						}

				// ���Ĳ���С�������
				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						for(int p=0;p<TILE;p++)
					{
						cBlockOne[threadIdy][threadIdx] += aBlock[threadIdy][p] * bBlock[p][threadIdx];
					}
			
				// ���岽��С������˽���ۼӵ������
				for(int threadIdy=0;threadIdy<TILE;threadIdy++)
					for(int threadIdx=0;threadIdx<TILE;threadIdx++)
						c[ cOffset + threadIdy*n + threadIdx] += cBlockOne[threadIdy][threadIdx];
			}
		}
	}
}
