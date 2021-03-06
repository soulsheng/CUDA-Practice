
#include <cuda_runtime.h>

#if 1 
typedef  float2 Point2;
#else
struct Point3
{
	float x,y,z;
};
#endif

void rgb2gray_cpu1( float* rgb, float* gray, int size );

void rgb2gray_cpu2( float* rgb, float* gray, int size );

void setArray( float* array, int size );

void printArray( float* array, int size ) ;
