

void warnup_gpu( float* rgb, float* gray, int size );

// GPU �汾1����������
void rgb2gray_gpu1( float* rgb, float* gray, int size );

// GPU �汾2���ϲ�
void rgb2gray_gpu2( float* rgb, float* gray, int size );
#if 0
// GPU �汾3��shared
void rgb2gray_gpu3( float* rgb, float* gray, int size );
#endif
