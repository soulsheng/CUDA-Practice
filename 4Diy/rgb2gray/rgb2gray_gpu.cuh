

void warnup_gpu( float* arrayA, float* arrayB, float* arrayC, int size );

// GPU �汾1����������
void rgb2gray_gpu1( float* arrayA, float* arrayB, float* arrayC, int size );

// GPU �汾2���ϲ�
void rgb2gray_gpu2( float* arrayA, float* arrayB, float* arrayC, int size );
#if 0
// GPU �汾3��shared
void rgb2gray_gpu3( float* arrayA, float* arrayB, float* arrayC, int size );
#endif
