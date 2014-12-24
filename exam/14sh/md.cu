#include <stdio.h>
#include "cuda_runtime.h"
#include "math.h"
#include "time.h"
#include "helper_timer.h"

//�������32bit���������p������n
__host__ void cpu_gen_rand(int *p, int n)
{
	srand(time(0));

	for (int i=0; i<n; i++)
		p[i] = rand();
}

//����ת�����ͣ����pf������Դpi������n
__host__ void cpu_int2float(float *pf, const int *pi, int n)
{
	for (int i=0; i<n; i++)
		pf[i] = float(pi[i]);
}

__global__ void kernel_int2float(float *pf, const int *pi)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	pf[index] = float(pi[index]);
}

__host__ void gpu_int2float(float *pf, const int *pi, int n)
{
	int blockDim = 1024;
	int gridDim = n/blockDim;
	kernel_int2float<<<gridDim, blockDim>>>(pf, pi);
}

//�����R*RT��p���������q��Դ����R������n
__host__ void cpu_matrix(float *p, const float *q, int n)
{
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<n; j++)
		{
			p[i*n+j] = q[i]*q[j];
		}
	}
}

#define TILE_DIM 16
#define BLOCK_DIM 1024

__global__ void kernel_matrix(float *p, const float *q, int n)
{
	int y = blockIdx.y*TILE_DIM+threadIdx.y;
	int x = blockIdx.x*TILE_DIM+threadIdx.x;
	p[(y)*n+(x)] = q[y]*q[x];
}

__host__ void gpu_matrix(float *p, const float *q, int n)
{
	int blockDim = 1024;
	int gridDim = n/blockDim;
	kernel_matrix<<<gridDim, blockDim>>>(p, q, n);
}

//��ģ||R||��p������R������n
__host__ float cpu_abs(const float *p, int n)
{
	float sum = 0.0f;
	for (int i=0; i<n; i++)
		sum += p[i]*p[i];
	return sqrt(sum);
}

__global__ void kernel_abs(float *pc, const float *p)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float vec[BLOCK_DIM];

	vec[threadIdx.x] = p[threadIdx.x];

	vec[threadIdx.x] *= vec[threadIdx.x];

	__syncthreads();

	int maxIndex = blockDim.x/2;

	while (maxIndex >= 1)
	{
		if (threadIdx.x < maxIndex)
			vec[threadIdx.x] += vec[threadIdx.x+maxIndex];

		maxIndex /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0)
		pc[blockIdx.x] = vec[0];
}

__host__ float gpu_abs(float *p, int n)
{
	float *dev_c, *hst_c;
	int blockDim = 1024;
	int gridDim = n/blockDim;

	cudaHostAlloc(&hst_c, gridDim*sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_c, gridDim*sizeof(float));

	kernel_abs<<<gridDim, blockDim>>>(dev_c, p);

	cudaMemcpy(hst_c, dev_c, gridDim*sizeof(float), cudaMemcpyDeviceToHost);

	float sum = 0.0f;
	for (int i=0; i<gridDim; i++)
		sum += hst_c[i];

	cudaFree(dev_c);
	cudaFreeHost(hst_c);

	return sum;
}

//��a����p�Ǿ���R*RT�����w���߶�h��r��ģ||R||��ԭλ����
__host__ void cpu_a(float *p, int w, int h, float r)
{
	int n = w*h;
	for (int i=0; i<n; i++)
		p[i] /= r;
}

__global__ void kernel_a(float *p, float r)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	p[index] /= r;
}

__host__ void gpu_a(float *p, int w, int h, float r)
{
	int blockDim = 1024;
	int gridDim = w*h/blockDim;
	kernel_a<<<gridDim, blockDim>>>(p, r);
}

//�����ma����Сֵmi��p�Ǿ���a�����ma��mi
__host__ void cpu_mami(float *pma, float *pmi, const float *p, int w, int h)
{
	float ma = p[0];
	float mi = p[0];
	int n = w*h;
	for (int i=0; i<n; i++)
	{
		if (p[i] > ma)
			ma = p[i];
		if (p[i] < mi)
			mi = p[i];
	}
	*pma = ma;
	*pmi = mi;
}

__global__ void kernel_mi(float *pc, const float *p)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float vec[BLOCK_DIM];

	vec[threadIdx.x] = p[threadIdx.x];

	__syncthreads();

	int maxIndex = blockDim.x/2;

	while (maxIndex >= 1)
	{
		if (threadIdx.x < maxIndex)
		{
			if (vec[threadIdx.x] > vec[threadIdx.x+maxIndex])
				vec[threadIdx.x] = vec[threadIdx.x+maxIndex];
		}

		maxIndex /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0)
		pc[blockIdx.x] = vec[0];
}

__global__ void kernel_ma(float *pc, const float *p)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float vec[BLOCK_DIM];

	vec[threadIdx.x] = p[threadIdx.x];

	__syncthreads();

	int maxIndex = blockDim.x/2;

	while (maxIndex >= 1)
	{
		if (threadIdx.x < maxIndex)
		{
			if (vec[threadIdx.x] < vec[threadIdx.x+maxIndex])
				vec[threadIdx.x] = vec[threadIdx.x+maxIndex];
		}

		maxIndex /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0)
		pc[blockIdx.x] = vec[0];
}

__host__ void gpu_mami(float *pma, float *pmi, const float *p, int w, int h)
{
	float *dev_c, *hst_c;
	int blockDim = 1024;
	int gridDim = w*h/blockDim;

	cudaHostAlloc(&hst_c, gridDim*sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_c, gridDim*sizeof(float));

	kernel_ma<<<gridDim, blockDim>>>(dev_c, p);

	cudaMemcpy(hst_c, dev_c, gridDim*sizeof(float), cudaMemcpyDeviceToHost);

	float sum = 0.0f;
	float ma, mi;
	ma = hst_c[0];
	for (int i=0; i<gridDim; i++)
	{
		if (hst_c[i] > ma)
			ma = hst_c[i];
	}

	kernel_ma<<<gridDim, blockDim>>>(dev_c, p);
	cudaMemcpy(hst_c, dev_c, gridDim*sizeof(float), cudaMemcpyDeviceToHost);

	mi = hst_c[0];
	for (int i=0; i<gridDim; i++)
	{
		if (hst_c[i] < mi)
			mi = hst_c[i];
	}

	*pma = ma;
	*pmi = mi;

	cudaFree(dev_c);
	cudaFreeHost(hst_c);
}

__host__ void cpu_run(float *pm, float *pma, float*pmi, const int *pi, int n)
{
	float *pf = (float*)malloc(n*sizeof(float));
	StopWatchInterface *wt;
	sdkCreateTimer(&wt);

	//ת����������
	sdkStartTimer(&wt);
	cpu_int2float(pf, pi, n);
	sdkStopTimer(&wt);
	printf("cpu_int2float : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�����R*RT
	sdkStartTimer(&wt);
	cpu_matrix(pm, pf, n);
	sdkStopTimer(&wt);
	printf("cpu_matrix : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//��ģ||R||
	sdkStartTimer(&wt);
	float r = cpu_abs(pf, n);
	sdkStopTimer(&wt);
	printf("cpu_abs : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�����A����������pm�У�����n*n
	sdkStartTimer(&wt);
	cpu_a(pm, n, n, r);
	sdkStopTimer(&wt);
	printf("cpu_a : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�������ma��Сֵmi
	sdkStartTimer(&wt);
	cpu_mami(pma, pmi, pm, n, n);
	sdkStopTimer(&wt);
	printf("cpu_mami : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	free(pf);
}

__host__ void gpu_run(float *pm, float *pma, float*pmi, const int *pi, int n)
{
	cudaSetDevice(0);

	float *dev_f, *dev_m, *hst_m;
	int *dev_i, *hst_i;
	StopWatchInterface *wt;
	sdkCreateTimer(&wt);

	cudaMalloc((void**)&dev_i, n*sizeof(int));
	cudaMalloc((void**)&dev_f, n*sizeof(float));
    cudaMalloc((void**)&dev_m, n*n*sizeof(float));

	cudaHostAlloc(&hst_i, n*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&hst_m, n*n*sizeof(float), cudaHostAllocDefault);

	//��ҳ�ڴ�->��ҳ�ڴ�
	memcpy(hst_i, pi, n*sizeof(int));

	//��ҳ�ڴ�->�豸�ڴ�
	cudaMemcpy(dev_i, hst_i, n*sizeof(float), cudaMemcpyHostToDevice);

	//ת����������
	sdkStartTimer(&wt);
	gpu_int2float(dev_f, dev_i, n);
	sdkStopTimer(&wt);
	printf("gpu_int2float : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�����R*RT
	sdkStartTimer(&wt);
	gpu_matrix(dev_m, dev_f, n);
	sdkStopTimer(&wt);
	printf("gpu_matrix : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//��ģ||R||
	sdkStartTimer(&wt);
	float r = gpu_abs(dev_f, n);
	sdkStopTimer(&wt);
	printf("gpu_abs : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�����A����������pm�У�����n*n
	sdkStartTimer(&wt);
	gpu_a(dev_m, n, n, r);
	sdkStopTimer(&wt);
	printf("gpu_a : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�������ma��Сֵmi
	sdkStartTimer(&wt);
	gpu_mami(pma, pmi, dev_m, n, n);
	sdkStopTimer(&wt);
	printf("gpu_mami : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//�豸�ڴ�->��ҳ�ڴ�
	cudaMemcpy(hst_m, dev_m, n*n*sizeof(float), cudaMemcpyDeviceToHost);

	//��ҳ�ڴ�->��ҳ�ڴ�
	memcpy(pm, hst_m, n*n*sizeof(float));
	
	cudaFree(dev_i);
    cudaFree(dev_f);
    cudaFree(dev_m);

	cudaFreeHost(hst_i);
	cudaFreeHost(hst_m);

	cudaDeviceReset();
}

int main()
{
	int n = 8192;
	int *pi = (int*)malloc(n*sizeof(int));
	float *pm = (float*)malloc(n*n*sizeof(float));
	float ma, mi;
	StopWatchInterface *wt;
	sdkCreateTimer(&wt);

	//���������
	sdkStartTimer(&wt);
	cpu_gen_rand(pi, n);
	sdkStopTimer(&wt);
	printf("cpu_gen_rand : %3.1f ms\n", sdkGetTimerValue(&wt));
	sdkResetTimer(&wt);

	//cpu���У�pmΪ�������ma���ֵ��mi��Сֵ��pi���������������n����
	cpu_run(pm, &ma, &mi, pi, n);

	//gpu���У�pmΪ�������ma���ֵ��mi��Сֵ��pi���������������n����
	gpu_run(pm, &ma, &mi, pi, n);
	
	free(pi);
	free(pm);

	system("pause");

    return 0;
}
