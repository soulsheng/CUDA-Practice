
#include <iostream>
using namespace std;

// Step 3: 函数声明
void polygonDistance( float x0, float y0, float *xi, float *yi, int n, float *di );


void main()
{
	cout << "Polygon Distance begin: " << endl;

	// C编程模型：1.变量定义、2.数组申请、3.函数声明、4.函数实现、5.函数调用

	cout << "Step 1: 变量定义" << endl;
	
	// 一点 p0(x0,y0)
	float x0 = 1.0f;
	float y0 = 2.0f;

	int n = 10000;

	cout << "Step 2: 数组申请" << endl;

	// 多边形顶点 pi(xi,yi)
	float *xi = (float*)malloc( n * sizeof(float) );
	float *yi = (float*)malloc( n * sizeof(float) );
	float *di = (float*)malloc( n * sizeof(float) );

	for (int i=0;i<n;i++)
	{
		xi[i] = sin( (float)i/n );
		yi[i] = cos( (float)i/n );
		di[i] = 0.0f;
	}

	cout << "Step 5: 函数调用" << endl;
	polygonDistance( x0, y0, xi, yi, n, di);

	cout << "Polygon Distance end! " << endl;
}

// Step 4: 函数实现
void polygonDistance( float x0, float y0, float *xi, float *yi, int n, float *di )
{
	cout << "Step 3: 函数声明" << endl;
	cout << "Step 4: 函数实现" << endl;

	for ( int i=0; i<n; i++ )
	{
		di[i] = sqrt( (xi[i]-x0)*(xi[i]-x0) + (yi[i]-y0)*(yi[i]-y0) );
	}
}