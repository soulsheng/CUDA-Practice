
#include <iostream>
using namespace std;

// Step 3: 函数声明
void polygonDistance( float x0, float y0, float *x, float *y, int n, float *d );


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
	float *x = (float*)malloc( n * sizeof(float) );
	float *y = (float*)malloc( n * sizeof(float) );
	float *d = (float*)malloc( n * sizeof(float) );

	for (int i=0;i<n;i++)
	{
		x[i] = sin( (float)i/n );
		y[i] = cos( (float)i/n );
		d[i] = 0.0f;
	}

	cout << "Step 5: 函数调用" << endl;
	polygonDistance( x0, y0, x, y, n, d);

	cout << "Polygon Distance end! " << endl;
}

// Step 4: 函数实现
void polygonDistance( float x0, float y0, float *x, float *y, int n, float *d )
{
	cout << "Step 3: 函数声明" << endl;
	cout << "Step 4: 函数实现" << endl;

	for ( int i=0; i<n; i++ )
	{
		d[i] = sqrt( (x[i]-x0)*(x[i]-x0) + (y[i]-y0)*(y[i]-y0) );
	}
}