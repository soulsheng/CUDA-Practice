

void warnup_gpu( float* array, int size, int width, int r=1 );

// GPU �汾1s ֱ���ۼ� �д���
float boxfilter_gpu1s( float* array, int size, int width, int r=1 );

// GPU �汾1p ֱ���ۼ� �д���
float boxfilter_gpu1p( float* array, int size, int width, int r=1 );

// GPU �汾2s scan����� �д���
float boxfilter_gpu2s( float* array, int size, int width, int r=1 );

// GPU �汾2p scan����� �в���
float boxfilter_gpu2p( float* array, int size, int width, int r=1 );
