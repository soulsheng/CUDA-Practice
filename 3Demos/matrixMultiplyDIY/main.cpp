#include <iostream>
using namespace std;
#include "stdio.h"

#include "timerTest.h"
#include "matrixMultiplyCPU.h"
#include "matrixMultiplyGPU.cuh"


#define  MATRIX_WIDTH	512

void printMatrix( float* m, int n )
{
	if( n>16 )	return;
	// �ߴ�С��16*16ʱ���������������֤�����汾����Ƿ�һ�� 
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
			cout << m[ i*n+j ] << " " ;
		cout << endl ;
	}
		cout << endl << endl;

}

int main()
{
	timerTestCU  timerCPU;

	int nSize = MATRIX_WIDTH; 
	float *aMatrix , *bMatrix, *cMatrix;
	aMatrix = (float*)malloc( nSize*nSize*sizeof(float) );
	bMatrix = (float*)malloc( nSize*nSize*sizeof(float) );
	cMatrix = (float*)malloc( nSize*nSize*sizeof(float) );

	for(int i=0;i<nSize*nSize;i++)
	{
		aMatrix[i] = rand() % 10;
		bMatrix[i] = rand() % 10;	
	}
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );

	printMatrix( aMatrix, nSize );
	printMatrix( bMatrix, nSize );

	timerCPU.start();
	// CPU �汾1����ʼ
	cout <<"\n" <<  "CPU �汾1����ʼ" << endl;
	matrixMul1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU �汾2��blockԤ���ֿ�
	cout << "\n" << "CPU �汾2��blockԤ���ֿ�" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU �汾3��block�ֿ�
	cout << "\n" << "CPU �汾3��block�ֿ�" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	// CUDAԤ��
	cout << "\nCUDAԤ��" << endl;
	setupCUDA();
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );


#if 1
	timerCPU.start();
	// GPU �汾1����ʼ
	cout << "\n" << "GPU �汾1����ʼ" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize, true );
#endif

#if 1
	timerCPU.start();
	// GPU �汾2��block�ֿ飬DIY
	cout << "\n" << "GPU �汾2��block�ֿ飬DIY" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize, true );
#endif


#if 1
	timerCPU.start();
	// GPU �汾3��block�ֿ飬SDK
	cout << "\n" << "GPU �汾3��block�ֿ飬SDK" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize, true );
#endif


#if 1
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize ); // cuda BLAS ������ʼ��

	timerCPU.start();
	// GPU �汾4��cuda BLAS
	cout << "\n" << "GPU �汾4��cuda BLAS" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize, true );
#endif
}