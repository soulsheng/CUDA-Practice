

//--CPU 版本1：直接累加----------	
float reduction_cpu1( float* array, int size );

//--CPU 版本2：跨步长累加，步长递增----------	
float reduction_cpu2( float* array, int size );

//--CPU 版本3：跨步长累加，步长递减----------	
float reduction_cpu3( float* array, int size );



void setArray( float* array, int size );

void printArray( float* array, int size ) ;
