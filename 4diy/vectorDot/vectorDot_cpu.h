
#include <cuda_runtime.h>

#if 1 
typedef  float2 Point2;
#else
struct Point3
{
	float x,y,z;
};
#endif

void vectorDot_cpu1( float* arrayA, float* arrayB, float* arrayC, int size );

void setArray( float* array, int size );

void printArray( float* array, int size ) ;
