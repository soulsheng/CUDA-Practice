
// GPUԤ��
void setupCUDA();

// GPU �汾1����ʼ
void matrixMulGPU1( float* a, float*b, float*c, int n, bool bTimeKernel = false );

// GPU �汾2��block�ֿ飬DIY
void matrixMulGPU2( float* a, float*b, float*c, int n, bool bTimeKernel = false );

// GPU �汾3��block�ֿ飬SDK
void matrixMulGPU3( float* a, float*b, float*c, int n, bool bTimeKernel = false );
