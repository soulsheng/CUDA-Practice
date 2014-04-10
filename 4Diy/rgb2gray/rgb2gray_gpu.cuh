

void warnup_gpu( float* rgb, float* gray, int size );

// GPU 版本1：步长递增
void rgb2gray_gpu1( float* rgb, float* gray, int size );

// GPU 版本2：合并
void rgb2gray_gpu2( float* rgb, float* gray, int size );
#if 0
// GPU 版本3：shared
void rgb2gray_gpu3( float* rgb, float* gray, int size );
#endif
