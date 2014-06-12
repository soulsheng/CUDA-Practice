
// GPU预热
void setupCUDA();

// GPU 版本1，初始
void matrixMulGPU1( float* a, float*b, float*c, int n, bool bTimeKernel = false );

// GPU 版本2，block分块，DIY
void matrixMulGPU2( float* a, float*b, float*c, int n, bool bTimeKernel = false );

// GPU 版本3，block分块，SDK
void matrixMulGPU3( float* a, float*b, float*c, int n, bool bTimeKernel = false );
