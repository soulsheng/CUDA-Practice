

//--CPU 版本2：scan再相减----------	
float boxfilter_cpu2( float* array, int size, int width, int r=1 );

//--CPU 版本1：直接累加----------	
float boxfilter_cpu1( float* array, int size, int width, int r=1 );

void setArray( float* array, int size );

void printArray( float* array, int size, int width ) ;
