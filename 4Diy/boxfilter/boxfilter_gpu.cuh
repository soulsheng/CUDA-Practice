

void warnup_gpu( float* array, int size, int width );


// GPU �汾1 �д���
float boxfilter_gpu1( float* array, int size, int width );

// GPU �汾2 shared
float boxfilter_gpu2( float* array, int size, int width );
