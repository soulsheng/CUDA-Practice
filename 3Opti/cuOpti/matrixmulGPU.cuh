
// GPUԤ��
void setupCUDA();

// GPU �汾1����ά����
void matrixMulGPU1( float* a, float*b, float*c, int n );

// GPU �汾2��һά����
void matrixMulGPU2( float* a, float*b, float*c, int n );

// GPU �汾3��block�ֿ飬��ά����
void matrixMulGPU3( float* a, float*b, float*c, int n );

// GPU �汾4��block�ֿ飬��ά���� �Ż�, cBlockOne array2one
void matrixMulGPU4( float* a, float*b, float*c, int n );

// GPU �汾5��block�ֿ飬��ά���� �Ż�, ����
void matrixMulGPU5( float* a, float*b, float*c, int n );
