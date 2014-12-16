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
	// 尺寸小于16*16时，输出矩阵结果，验证各个版本结果是否一致 
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
			cout << m[ i*n+j ] << " " ;
		cout << endl ;
	}
		cout << endl << endl;

}

bool verify( float* m, float* m_ref, int n )
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if( m[ i*n+j ] != m_ref[ i*n+j ] )
				return false;
		}
	}
		
	return true; 
}

int main()
{
	timerTestCU  timerCPU;

	int nSize = MATRIX_WIDTH; 
	float *aMatrix , *bMatrix, *cMatrix, *cMatrix_ref;
	aMatrix = (float*)malloc( nSize*nSize*sizeof(float) );
	bMatrix = (float*)malloc( nSize*nSize*sizeof(float) );
	cMatrix = (float*)malloc( nSize*nSize*sizeof(float) );
	cMatrix_ref = (float*)malloc( nSize*nSize*sizeof(float) );

	for(int i=0;i<nSize*nSize;i++)
	{
		aMatrix[i] = rand() % 10;
		bMatrix[i] = rand() % 10;	
	}
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	memset( cMatrix_ref, 0, nSize*nSize*sizeof(float) );

	printMatrix( aMatrix, nSize );
	printMatrix( bMatrix, nSize );

	timerCPU.start();
	// CPU 版本1，初始
	cout <<"\n" <<  "CPU 版本1，初始" << endl;
	matrixMul1( aMatrix, bMatrix, cMatrix_ref, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	timerCPU.start();
	// CPU 版本2，block预备分块
	cout << "\n" << "CPU 版本2，block预备分块" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul2( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	if( verify( cMatrix, cMatrix_ref, nSize ) )
		cout << "CPU 版本2，verify passed!" << endl;
	else
		cout << "CPU 版本2，verify failed!" << endl;

	timerCPU.start();
	// CPU 版本3，block分块
	cout << "\n" << "CPU 版本3，block分块" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMul3( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;

	if( verify( cMatrix, cMatrix_ref, nSize ) )
		cout << "CPU 版本3，verify passed!" << endl;
	else
		cout << "CPU 版本3，verify failed!" << endl;

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
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
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
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
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
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU3( aMatrix, bMatrix, cMatrix, nSize, true );
#endif


#if 1
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize ); // cuda BLAS 环境初始化

	timerCPU.start();
	// GPU 版本4，cuda BLAS
	cout << "\n" << "GPU 版本4，cuda BLAS" << endl;
	memset( cMatrix, 0, nSize*nSize*sizeof(float) );
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize );
	printMatrix( cMatrix, nSize );
	timerCPU.stop();
	cout << "Total time(ms) : " << timerCPU.getTime() << endl;
	matrixMulGPU4( aMatrix, bMatrix, cMatrix, nSize, true );
#endif
}