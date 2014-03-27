
#include "iostream"
using namespace std;

#include "utilityTimer.h"
#include "multiplymatrixCPU.h"

#define NMATRIX	4
#define PRINT	1


extern "C"	
	int  multiplymatrixGPU( float* a, float* b, float* c, int n );

void printMatrix(float* m, int n)
{
#if PRINT
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			cout<< m[i*n +j] << " ";
		}
		cout << "\n";
	}
	cout << "\n";
#endif
}

int main()
{
	TimerCPU	timerC;

	int nMatrix = NMATRIX;
	float *aMatrix, *bMatrix, *cMatrix;
	aMatrix = (float*)malloc( nMatrix*nMatrix*sizeof(float) );
	bMatrix = (float*)malloc( nMatrix*nMatrix*sizeof(float) );
	cMatrix = (float*)malloc( nMatrix*nMatrix*sizeof(float) );

	for (int i=0;i<nMatrix*nMatrix;i++)
	{
		aMatrix[i]= rand()%10;//(float)RAND_MAX;
		bMatrix[i]= rand()%10;//(float)RAND_MAX;
		cMatrix[i]= 0.0f;
	}
	
	printMatrix(aMatrix,nMatrix);
	printMatrix(bMatrix,nMatrix);
	printMatrix(cMatrix,nMatrix);

	float timeValueC, timeValueG;
	timerC.start();
	multiplymatrixCPU2( aMatrix, bMatrix, cMatrix, nMatrix );
	timerC.stop();
	timeValueC = timerC.getTime();
	cout<< "time is "<< timeValueC << " ms" << endl ;

	printMatrix(cMatrix,nMatrix);

	timerC.start();
	multiplymatrixGPU( aMatrix, bMatrix, cMatrix, nMatrix );
	timerC.stop();
	timeValueG = timerC.getTime();
	cout<< "time is "<< timeValueG << " ms" << endl ;

	cout<< "speedup is "<< timeValueC/timeValueG << endl;

	printMatrix(cMatrix,nMatrix);

	return 0;
}