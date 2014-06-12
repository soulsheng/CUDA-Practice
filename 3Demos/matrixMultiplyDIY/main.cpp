#include <iostream>
using namespace std;
#include "stdio.h"

#include "timerCPP.h"
#include "matrixMultiplyCPU.h"
#include "matrixMultiplyGPU.cuh"
#include "timerCUDA.h"

#define  MATRIX_WIDTH	512

void printMatrix( float* m, int n )
{
	if( n>16 )	return;
	// 尺寸小于16*16时，输出矩阵结果，验证各个版本结果是否一致 
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
	// CPU 版本1，初始
	cout <<"\n" <<  "CPU 版本1，初始" << endl;
	matrixMul1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 版本2，block预备分块
	cout << "\n" << "CPU 版本2，block预备分块" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 版本3，block分块
	cout << "\n" << "CPU 版本3，block分块" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << timerCPU.getTime() << endl;

	// CUDA预热
	cout << "\nCUDA预热" << endl;
	setupCUDA();
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );


#if 1
	timerCPU.start();
	// GPU 版本1，初始
	cout << "\n" << "GPU 版本1，初始" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time : " << timerCPU.getTime() << endl;
	matrixMulGPU1( aMatrix, bMatrix, cMatrix, nSize, true );
#endif

#if 1
	timerCPU.start();
	// GPU 版本2，block分块，DIY
	cout << "\n" << "GPU 版本2，block分块，DIY" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time : " << timerCPU.getTime() << endl;
	matrixMulGPU2( aMatrix, bMatrix, cMatrix, nSize, true );
#endif


#if 1
	timerCPU.start();
	// GPU 版本3，block分块，SDK
	cout << "\n" << "GPU 版本3，block分块，SDK" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time : " << timerCPU.getTime() << endl;
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize, true );
#endif

}