
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h> 

#define VECTOR_LEN	(2048)
#define BLOCK_DIM	(256)

void cpu_calculate(int	*vetor,
				   int   vector_len,
				   float *min,
				   float *max,
				   float *mat)
{
	int   i, j;
	long long result;
	float zhi = 0.0f;
	float max_tmp, min_tmp,  tmp;

	result = 0;
	for(i = 0; i < vector_len; i++)
	{
		result += vetor[i] * vetor[i];
	}
	zhi = sqrt((double) result);
	printf("cpu zhi:%f\n", zhi);

	max_tmp = -10000.0f;
	min_tmp = 100000.0f;
	for(i = 0; i < vector_len; i++)
	{
		for(j = 0; j < vector_len; j++)
		{
			tmp = vetor[i] * vetor[j] * 1.0f / zhi;

			if(tmp > max_tmp)
			{
				max_tmp = tmp;
			}

			if(tmp < min_tmp)
			{
				min_tmp = tmp;
			}

			mat[i*vector_len+j] = tmp;
		}
	}

	*min = min_tmp;
	*max = max_tmp;
}

__global__ void calulate_max_min(float *mul_res,
								 float *max_res,
								 float *min_res)
{
	int x, i, step;
	float max_tmp, min_tmp, max, min;
	__shared__ float share_mem[BLOCK_DIM];

	x = blockDim.x * blockIdx.x + threadIdx.x;

	share_mem[threadIdx.x] = mul_res[x];

	__syncthreads();

	max_tmp = -10000.0f;
	min_tmp = 100000.0f;
	for(x = 1; x < BLOCK_DIM; x *= 2)
	{
		if(threadIdx.x % (2 * x) == 0)
		{
			max = share_mem[threadIdx.x] > share_mem[threadIdx.x + x] ? share_mem[threadIdx.x] : share_mem[threadIdx.x + x];
			min = share_mem[threadIdx.x] < share_mem[threadIdx.x + x] ? share_mem[threadIdx.x] : share_mem[threadIdx.x + x];

			if(max_tmp < max)
			{
				max_tmp = max;
			}

			if(min_tmp > min)
			{
				min_tmp = min;
			}
		}
	}

	if(threadIdx.x == 0)
	{
		max_res[blockIdx.x] = max_tmp;
		min_res[blockIdx.x] = min_tmp;
	}
}

__global__ void calculate_mul(int   *vetor,
							  int   vector_len,
							  float zhi,
							  float *res)
{
	int x, y;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;

	res[y * vector_len + x] = vetor[x] * vetor[y] * 1.0f / zhi;
}

__global__ void square_sum_kernel( int   *vetor,
								   int   vector_len,
								   int   *square_sum)
{
	int x, i, step;
	__shared__ int share_mem[BLOCK_DIM];

	x = blockDim.x * blockIdx.x + threadIdx.x;

	if(x < vector_len)
	{
		share_mem[threadIdx.x] = vetor[x] * vetor[x];
	}
	else
	{
		share_mem[threadIdx.x] = 0;
	}	

	__syncthreads();

	for(x = 1; x < BLOCK_DIM; x *= 2)
	{
		if(threadIdx.x % (2 * x) == 0)
		{
			share_mem[threadIdx.x] += share_mem[threadIdx.x + x];
		}
	}

	if(threadIdx.x == 0)
	{
		square_sum[blockIdx.x] = share_mem[0];
	}
}

void gpu_calculate(int	*vetor,
				   int   vector_len,
				   float *min,
				   float *max,
				   float *mat)
{
	int i, res;
	dim3  grid_dim, block_dim;
	cudaError_t cudaStatus;
	float zhi, max_tmp, min_tmp;
	int *vector_dev = NULL;
	int *square_sum = NULL;
	int *square_sum_cpu = NULL;
	float *mat_mul_res = NULL;
	float *max_res = NULL;
	float *min_res = NULL;
	float *max_res_cpu = NULL;
	float *min_res_cpu = NULL;

	cudaStatus = cudaMalloc((void**)&mat_mul_res, vector_len * vector_len * sizeof(float));
	if (cudaStatus != cudaSuccess || mat_mul_res == NULL) 
	{
		printf("cudaMalloc failed\n");
        goto INIT_FAILED;
    }

	cudaStatus = cudaMalloc((void**)&vector_dev, vector_len * sizeof(int));
	if (cudaStatus != cudaSuccess || vector_dev == NULL) 
	{
		printf("cudaMalloc failed\n");
        goto INIT_FAILED;
    }

	cudaStatus = cudaMalloc((void**)&square_sum, sizeof(int) * (vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (cudaStatus != cudaSuccess || square_sum == NULL) 
	{
		printf("cudaMalloc failed\n");
        goto INIT_FAILED;
    }

	cudaStatus = cudaMalloc((void**)&max_res, sizeof(float) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (cudaStatus != cudaSuccess || max_res == NULL) 
	{
		printf("cudaMalloc failed\n");
        goto INIT_FAILED;
    }

	cudaStatus = cudaMalloc((void**)&min_res, sizeof(float) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (cudaStatus != cudaSuccess || min_res == NULL) 
	{
		printf("cudaMalloc failed\n");
        goto INIT_FAILED;
    }

	cudaStatus = cudaMemcpy(vector_dev, vetor, vector_len * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
	{
        goto INIT_FAILED;
    }

	square_sum_cpu = (int*) malloc (sizeof(int) * (vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (square_sum_cpu == NULL) 
	{
        goto INIT_FAILED;
    }

	max_res_cpu = (float*) malloc (sizeof(float) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (max_res_cpu == NULL) 
	{
        goto INIT_FAILED;
    }

	min_res_cpu = (float*) malloc (sizeof(float) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM);
	if (min_res_cpu == NULL) 
	{
        goto INIT_FAILED;
    }

	block_dim.x = BLOCK_DIM;
	block_dim.y = 1;

	grid_dim.x = (vector_len + BLOCK_DIM - 1) / BLOCK_DIM; 
	grid_dim.y = 1;

	square_sum_kernel<<<grid_dim, block_dim>>>(vector_dev, vector_len, square_sum);

	cudaStatus = cudaMemcpy(square_sum_cpu, square_sum, sizeof(int) * (vector_len + BLOCK_DIM - 1) / BLOCK_DIM, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
        goto INIT_FAILED;
    }

	res = 0;
	for(i = 0; i < (vector_len + BLOCK_DIM - 1) / BLOCK_DIM; i++)
	{
		res += square_sum_cpu[i];
	}

	zhi = sqrt((double)res);
	printf("gpu zhi:%f\n", zhi);

	block_dim.x = 16;
	block_dim.y = 16;

	grid_dim.x = (vector_len + 16 - 1) / 16; 
	grid_dim.y = (vector_len + 16 - 1) / 16;

	calculate_mul<<<grid_dim, block_dim>>>(vector_dev, vector_len, zhi, mat_mul_res);

	cudaMemcpy( mat, mat_mul_res, sizeof(int) * vector_len * vector_len, cudaMemcpyDeviceToHost );

	block_dim.x = BLOCK_DIM;
	block_dim.y = 1;

	grid_dim.x = (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM; 
	grid_dim.y = 1;

	calulate_max_min<<<grid_dim, block_dim>>>(mat_mul_res, max_res, min_res);

	cudaStatus = cudaMemcpy(max_res_cpu, max_res, sizeof(int) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
        goto INIT_FAILED;
    }

	cudaStatus = cudaMemcpy(min_res_cpu, min_res, sizeof(int) * (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
        goto INIT_FAILED;
    }

	max_tmp = -100000.0f;
	min_tmp =  100000.0f;
	for(i = 0; i < (vector_len * vector_len + BLOCK_DIM - 1) / BLOCK_DIM; i++)
	{
		if(max_tmp < max_res_cpu[i])
		{
			max_tmp = max_res_cpu[i];
		}

		if(min_tmp > min_res_cpu[i])
		{
			min_tmp = min_res_cpu[i];
		}
	}

	*max = max_tmp;
	*min = min_tmp;

INIT_FAILED:
	if(mat_mul_res != NULL)
	{
		cudaFree(mat_mul_res);
		mat_mul_res = NULL;
	}

	if(vector_dev != NULL)
	{
		cudaFree(vector_dev);
		vector_dev = NULL;
	}

	if(square_sum_cpu != NULL)
	{
		free(square_sum_cpu);
		square_sum_cpu = NULL;
	}
}

bool verify(float *ab,float *abGpu,int n)
{
	int err = 0;
	for(int i=0;i<n;i++){
		if(ab[i]!=0 && fabs(ab[i]-abGpu[i])/fabs(ab[i])>1e-5){
			err++;//return false;
		}
	}
	if(err)
		return false;
	else
		return true;
}

int main()
{
	int i;
	int	*vector = NULL;
	float cpu_min, cpu_max, gpu_min, gpu_max;

	vector		= (int*) malloc (sizeof(int) * VECTOR_LEN);
	if(vector == NULL)
	{
		printf("malloc failed\n");
		goto INIT_FAILED;
	}

	for(i = 0; i < VECTOR_LEN; i++)
	{
		vector[i] = rand() % 10;
	}

	int n = VECTOR_LEN;
	float *mat = (float*)malloc(n*n*sizeof(float));
	float *mat_ref = (float*)malloc(n*n*sizeof(float));

	cpu_calculate(vector, VECTOR_LEN, &cpu_min, &cpu_max, mat_ref);
	gpu_calculate(vector, VECTOR_LEN, &gpu_min, &gpu_max, mat);

	printf("cpu max: %f, min:%f\n", cpu_max, cpu_min);
	printf("gpu max: %f, min:%f\n", gpu_max, gpu_min);
	
	cudaError err = cudaGetLastError();
	if( err!= cudaSuccess )
		printf( "failed \n" );

	bool verifySeccuss=verify( mat, mat_ref, n*n );

	if(verifySeccuss)
		printf("Verify Seccuss.\n");
	else
		printf("Verify Error!\n");

	free( mat );
	free( mat_ref );

INIT_FAILED:
	if(vector != NULL)
	{
		free(vector);
		vector = NULL;
	}

    return 0;
}
