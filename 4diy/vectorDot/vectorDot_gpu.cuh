

void warnup_gpu( float* arrayA, float* arrayB, float* arrayC, int size );

// GPU 版本1：步长递增
void vectorDot_gpu1( float* arrayA, float* arrayB, float* arrayC, int size );
#if 0
// GPU 版本2：步长递减
void vectorDot_gpu2( float* arrayA, float* arrayB, float* arrayC, int size );

// GPU 版本3：shared
void vectorDot_gpu3( float* arrayA, float* arrayB, float* arrayC, int size );
#endif
