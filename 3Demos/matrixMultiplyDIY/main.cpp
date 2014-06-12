#include <iostream>
using namespace std;
#include "stdio.h"

#include "timerCPP.h"
#include "matrixMultiplyCPU.h"
#include "matrixMultiplyGPU.cuh"
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
	// CPU 版本1，二维索引
	cout <<"\n" <<  "CPU 版本1，二维索引" << endl;
	matrixMul1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 版本2，一维索引
	cout << "\n" << "CPU 版本2，一维索引" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 版本3，block分块，二维索引
	cout << "\n" << "CPU 版本3，block分块，二维索引" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	// 预热
	cout << "\n预热" << endl;
	setupCUDA();
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );

#if 1
	timerCPU.start();
	// GPU 版本2，一维索引
	cout << "\n" << "GPU 版本2，一维索引" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "CPU time : " << timerCPU.getTime() << endl;	
#endif

#if 1
	timerCPU.start();
	// GPU 版本1，一维索引
	cout << "\n" << "GPU 版本1，二维索引" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "CPU time : " << timerCPU.getTime() << endl;
#endif

#if 1
	timerCPU.start();
	// GPU 版本3，block 初步
	cout << "\n" << "GPU 版本3，block初步" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "CPU time : " << timerCPU.getTime() << endl;
#endif


#if 1
	timerCPU.start();
	// GPU 版本4，block SAVE memory
	cout << "\n" << "GPU 版本4，block SAVE memory" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "CPU time : " << timerCPU.getTime() << endl;
#endif


#if 1
	timerCPU.start();
	// GPU 版本5，block 分块，步长
	cout << "\n" << "GPU 版本5，block 分块，步长" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU5( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "CPU time : " << timerCPU.getTime() << endl;
#endif

}