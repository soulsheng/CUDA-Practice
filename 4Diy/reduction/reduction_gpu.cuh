

void warnup_gpu( float* array, int size );

// GPU 版本1：步长递增
float reduction_gpu1( float* array, int size );

// GPU 版本2：步长递减
float reduction_gpu2( float* array, int size );

// GPU 版本3：shared
float reduction_gpu3( float* array, int size );
