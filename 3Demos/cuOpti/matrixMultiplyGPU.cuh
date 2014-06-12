
// GPU预热
void setupCUDA();

// GPU 版本1，二维索引
void matrixMulGPU1( float* a, float*b, float*c, int n );

// GPU 版本2，一维索引
void matrixMulGPU2( float* a, float*b, float*c, int n );

// GPU 版本3，block分块，二维索引
void matrixMulGPU3( float* a, float*b, float*c, int n );

// GPU 版本4，block分块，二维索引 优化, cBlockOne array2one
void matrixMulGPU4( float* a, float*b, float*c, int n );

// GPU 版本5，block分块，二维索引 优化, 步长
void matrixMulGPU5( float* a, float*b, float*c, int n );
