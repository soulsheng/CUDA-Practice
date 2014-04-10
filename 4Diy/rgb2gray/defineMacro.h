

#ifndef	__DEFINE_MACRO__
#define __DEFINE_MACRO__

#include <vector_types.h>


#define		BLOCKDIM	256
#define		R_RATIO	(0.299f)
#define		G_RATIO	(0.587f)
#define		B_RATIO	(0.114f)

#define		N		(1<<24)
#define		REPEAT   10

#define  PRINTMAX   1024

#if 1
typedef  float3 ColorRGB;
#else
struct __builtin_align__(16) ColorRGB
{
	float r,g,b;
};
#endif

#endif // __DEFINE_MACRO__