
	// 3重循环
	int  multiplymatrixCPU( float* a, float* b, float* c, int n );
	
	// 2重循环
	int  multiplymatrixCPU2( float* a, float* b, float* c, int n );

	// block 2*2
	int  multiplymatrixCPU3( float* a, float* b, float* c, int n );

	// block 16*16
	int  multiplymatrixCPU4( float* a, float* b, float* c, int n );
