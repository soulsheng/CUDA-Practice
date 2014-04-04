

void warnup_gpu( float* array, int size );


// GPU 版本1 行串行
float scan_gpu1( float* array, int size );

// GPU 版本2 shared
float scan_gpu2( float* array, int size );
