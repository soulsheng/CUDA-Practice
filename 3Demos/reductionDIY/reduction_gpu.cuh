

void warnup_gpu( float* array, int size );

// GPU �汾1���粽���ۼӣ���������
float reduction_gpu1( float* array, int size, bool bTimeKernel = false );

// GPU �汾2���粽���ۼӣ������ݼ�
float reduction_gpu2( float* array, int size, bool bTimeKernel = false );

// GPU �汾3��shared memory
float reduction_gpu3( float* array, int size, bool bTimeKernel = false );
