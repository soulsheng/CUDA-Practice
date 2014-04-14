

void warnup_gpu( float* array, int size, int width, int r=1 );

// GPU 版本1s 直接累加 行串行
float boxfilter_gpu1s( float* array, int size, int width, int r=1 );

// GPU 版本1p 直接累加 行串行
float boxfilter_gpu1p( float* array, int size, int width, int r=1 );

// GPU 版本2s scan再相减 行串行
float boxfilter_gpu2s( float* array, int size, int width, int r=1 );

// GPU 版本2p scan再相减 行并行
float boxfilter_gpu2p( float* array, int size, int width, int r=1 );
