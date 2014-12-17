
#include <iostream>
using namespace std;

// Step 3: ��������
void polygonDistance( float x0, float y0, float *xi, float *yi, int n, float *di );


void main()
{
	cout << "Polygon Distance begin: " << endl;

	// C���ģ�ͣ�1.�������塢2.�������롢3.����������4.����ʵ�֡�5.��������

	cout << "Step 1: ��������" << endl;
	
	// һ�� p0(x0,y0)
	float x0 = 1.0f;
	float y0 = 2.0f;

	int n = 10000;

	cout << "Step 2: ��������" << endl;

	// ����ζ��� pi(xi,yi)
	float *xi = (float*)malloc( n * sizeof(float) );
	float *yi = (float*)malloc( n * sizeof(float) );
	float *di = (float*)malloc( n * sizeof(float) );

	for (int i=0;i<n;i++)
	{
		xi[i] = sin( (float)i/n );
		yi[i] = cos( (float)i/n );
		di[i] = 0.0f;
	}

	cout << "Step 5: ��������" << endl;
	polygonDistance( x0, y0, xi, yi, n, di);

	cout << "Polygon Distance end! " << endl;
}

// Step 4: ����ʵ��
void polygonDistance( float x0, float y0, float *xi, float *yi, int n, float *di )
{
	cout << "Step 3: ��������" << endl;
	cout << "Step 4: ����ʵ��" << endl;

	for ( int i=0; i<n; i++ )
	{
		di[i] = sqrt( (xi[i]-x0)*(xi[i]-x0) + (yi[i]-y0)*(yi[i]-y0) );
	}
}