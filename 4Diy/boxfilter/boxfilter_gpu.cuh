

void warnup_gpu( float* array, int size, int width );


// GPU 版本1 行串行
float boxfilter_gpu1( float* array, int size, int width );

// GPU 版本2 shared
float boxfilter_gpu2( float* array, int size, int width );
