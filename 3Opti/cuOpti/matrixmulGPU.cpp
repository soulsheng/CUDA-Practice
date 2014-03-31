#include <iostream>
using namespace std;
#include "stdio.h"

#include "timerCPP.h"

extern "C" void matrixMul( float* a, float*b, float*c, int n );
extern "C" void matrixMul2( float* a, float*b, float*c, int n );
extern "C" void matrixMul3( float* a, float*b, float*c, int n );

extern "C" void matrixMulGPU( float* a, float*b, float*c, int n );

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
	timerC  timerGPU;

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
	// CPU 版本1
	cout <<"\n" <<  "CPU 版本1" << endl;
	matrixMul( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 一重循环
	cout << "\n" << "CPU 版本2 一重循环" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU block
	cout << "\n" << "CPU 版本3 block" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerGPU.start();
	// GPU 版本1
	cout << "\n" << "GPU 版本1" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerGPU.stop();
	cout << timerGPU.getTime() << endl;
}