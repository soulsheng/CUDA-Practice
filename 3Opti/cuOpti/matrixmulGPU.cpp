#include <iostream>
using namespace std;
#include "stdio.h"

#include "timerCPP.h"
#include "referCPP.h"
#include "matrixmulGPU.cuh"
#include "timerCUDA.h"

void printMatrix( float* m, int n )
{
	return;
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
	timerC  timerCPU;
	timerCUDA	timerGPU;

	int nSize = 512;
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
	// CPU �汾1����ά����
	cout <<"\n" <<  "CPU �汾1����ά����" << endl;
	matrixMul1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU �汾2��һά����
	cout << "\n" << "CPU �汾2��һά����" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU �汾3��block�ֿ飬��ά����
	cout << "\n" << "CPU �汾3��block�ֿ飬��ά����" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;
#if 1
	timerGPU.start();
	// GPU �汾2��һά����
	cout << "\n" << "GPU �汾2��һά����" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerGPU.stop();
	cout << timerGPU.getTime() << endl;
#endif
#if 1
	timerGPU.start();
	// GPU �汾1��һά����
	cout << "\n" << "GPU �汾1����ά����" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerGPU.stop();
	cout << timerGPU.getTime() << endl;
#endif

#if 1
	timerGPU.start();
	// GPU �汾3��block ����
	cout << "\n" << "GPU �汾3��block����" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerGPU.stop();
	cout << timerGPU.getTime() << endl;
#endif


#if 1
	timerGPU.start();
	// GPU �汾4��block �Ż�
	cout << "\n" << "GPU �汾4��block�Ż�" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerGPU.stop();
	cout << timerGPU.getTime() << endl;
#endif
}