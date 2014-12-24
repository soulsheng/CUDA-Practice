#include <stdio.h>
#include <assert.h>
#include <stdlib.h>   
#include <malloc.h>

#include <math.h>
#include <conio.h>
#include <cuda_runtime.h>

#include <memory.h> 
//
#define N 8192      //总计算规模
#define S_DIM 512    //计算模的分块
#define B_DIM 32    //计算矩阵值的分块

#define MAXMIN_B_DIM 256
#define MAXMIN_DIM 1024  //算最大最小值的分块
bool verify(float *ab,float *abGpu,int n)
{
	for(int i=0;i<n;i++){
		if(ab[i]!=0 && fabs(ab[i]-abGpu[i])/fabs(ab[i])>1e-5){
			return false;
		}
	}
	return true;
}

//-----------------------cpu运行子函数---------------------------------
float   cpuAmpR(int *R)
{
   float sum=0.0;
   for(unsigned int  i=0;i<N;i++)
  {
	  float Rtmp=R[i];
      sum+=Rtmp*Rtmp;
   }
   return sqrtf(sum);
}


void cpuCountA(int *R,float Amp,float *A)
{
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int idx=i*N+j;
			float Ri=R[i];
			float Rj=R[j];
			A[idx]=Ri*Rj/Amp;
		}
	}
}
float cpuMaxA(float *A)
{
	float MaxA=A[0];
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int idx=i*N+j;
			if(A[idx]>MaxA){
				MaxA=A[idx];
			}
		}
	}
	return MaxA;
}
float cpuMinA(float *A)
{
	float MinA=A[0];
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int idx=i*N+j;
			if(A[idx]<MinA){
				MinA=A[idx];
			}
		}
	}
	return MinA;
}
//=============================================================

//-------------------------GPU核函数---------------------------

__global__   void  gpuAmpR(int *R,float *SumTemp)
{
	__shared__ float SData[S_DIM];
	int  bx=blockIdx.x;  
	int  tx=threadIdx.x;
	float Rtmp=R[bx*blockDim.x+tx];
	SData[tx]=Rtmp*Rtmp;

	__syncthreads();
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tx<s){
			SData[tx]+=SData[tx+s];
		}
		__syncthreads();
	}

	if(tx==0) SumTemp[bx]=SData[0];

}
__global__ void gpuCountA(int *R,float *Amp,float *A)
{
	int  bx=blockIdx.x;  
	int  by=blockIdx.y;
	int  tx=threadIdx.x;
	int  ty=threadIdx.y;
	int Idx;

	__shared__  float AS[B_DIM];
	__shared__  float BS[B_DIM];
	if(ty==0){
		if(bx*B_DIM+tx<N)   
			AS[tx] = R[bx*B_DIM+tx];  
		else  
			AS[tx] = 0;
	}
	if(ty==1){
		if(by*B_DIM+tx<N)   
			BS[tx] = R[by*B_DIM+tx];  
		else  
			BS[tx] = 0;
	}
	__syncthreads();
	Idx=(by*B_DIM+ty)*N+bx*B_DIM+tx;
	if(Idx<N*N){
		A[Idx]=AS[tx]*BS[ty]/(*Amp);
	}
}
__global__ void gpuMaxA(float *A,float *MaxTemp)
{
	__shared__ float SData[MAXMIN_DIM];
	__shared__ float max_Block;
	int  bx=blockIdx.x;  
	int  tx=threadIdx.x;
	

	int tid=bx*blockDim.x+tx;
	int Times=(N*N+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
	for(int i=0;i<Times;i++){
		if(tid<N*N){
			SData[tx]=A[tid];
		}
		else{
			SData[tx]=0;
		}
		__syncthreads();
		for(unsigned int s=blockDim.x/2;s>0;s>>=1){
			if(tx<s){
				if(SData[tx]<SData[tx+s]){
					SData[tx+s]=SData[tx];
				}
			}
			__syncthreads();
		}
		if(tx==0){
			if(i==0){
				max_Block=SData[0];
			}
			else{
				if(max_Block<SData[0]){
					max_Block=SData[0];
				}
			}
		}
		__syncthreads();
		tid+=gridDim.x*blockDim.x;
	}
	if(tx==0){
		MaxTemp[bx]=max_Block;
	}
}

__global__ void gpuMinA(float *A,float *MinTemp)
{
	__shared__ float SData[MAXMIN_DIM];
	__shared__ float min_Block;
	int  bx=blockIdx.x;  
	int  tx=threadIdx.x;
	

	int tid=bx*blockDim.x+tx;
	int Times=(N*N+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
	for(int i=0;i<Times;i++){
		if(tid<N*N){
			SData[tx]=A[tid];
		}
		else{
			SData[tx]=0;
		}
		__syncthreads();
		for(unsigned int s=blockDim.x/2;s>0;s>>=1){
			if(tx<s){
				if(SData[tx]>SData[tx+s]){
					SData[tx+s]=SData[tx];
				}
			}
			__syncthreads();
		}
		if(tx==0){
			if(i==0){
				min_Block=SData[0];
			}
			else{
				if(min_Block<SData[0]){
					min_Block=SData[0];
				}
			}
		}
		__syncthreads();
		tid+=gridDim.x*blockDim.x;
	}
	if(tx==0){
		MinTemp[bx]=min_Block;
	}
}



//============================================================


void main()
{
	int *data_R;
	float *SumR,SumR2,temp;
	float *Amp_R,*A_array,*A_min,*A_max,*MaxTemp,*MinTemp;//,*data_b,*data_c1,*data_c2;
	bool verifySeccuss=0;
	int blockNum=(N+S_DIM-1)/S_DIM;
	cudaMallocManaged(&data_R,N*sizeof(int));
	cudaMallocManaged(&Amp_R,sizeof(float));
	cudaMallocManaged(&SumR,blockNum*sizeof(float));
	cudaMallocManaged(&A_max,sizeof(float));
	cudaMallocManaged(&A_min,sizeof(float));
	cudaMallocManaged(&MaxTemp,MAXMIN_B_DIM*sizeof(float));
	cudaMallocManaged(&MinTemp,MAXMIN_B_DIM*sizeof(float));
	cudaMallocManaged(&A_array,N*N*sizeof(float));

	for(int i=0;i<N;i++){
		data_R[i]=rand();
	}

	*Amp_R=cpuAmpR(data_R);
	cpuCountA(data_R,*Amp_R,A_array);
	*A_max=cpuMaxA(A_array);
	*A_min=cpuMinA(A_array);



	printf("CPU 计算结果：\n|R| %f，最大值 %f，最小值 %f\n",*Amp_R,*A_max,*A_min);


	gpuAmpR<<<blockNum,S_DIM>>>(data_R,SumR);
	cudaDeviceSynchronize();
	SumR2=0.0;
	for (int i=0;i<blockNum;i++){
		SumR2+=SumR[i];
	}

	*Amp_R=sqrt(SumR2);
	dim3 mygrid(((N+B_DIM-1)/B_DIM),(N+B_DIM-1)/B_DIM);  
	dim3 myblock(B_DIM,B_DIM);  

	gpuCountA<<<mygrid,myblock>>>(data_R,Amp_R,A_array);


	gpuMaxA<<<MAXMIN_B_DIM,MAXMIN_DIM>>>(A_array,MaxTemp);
	gpuMinA<<<MAXMIN_B_DIM,MAXMIN_DIM>>>(A_array,MinTemp);

	cudaDeviceSynchronize();
	temp=MaxTemp[0];
	for(int i=1;i<MAXMIN_B_DIM;i++){
		if(temp<MaxTemp[i]){
			temp=MaxTemp[i];
		}
	}
	*A_max=temp;

	temp=MinTemp[0];
	for(int i=1;i<MAXMIN_B_DIM;i++){
		if(temp>MinTemp[i]){
			temp=MinTemp[i];
		}
	}
	*A_min=temp;

	*A_max=cpuMaxA(A_array);
	*A_min=cpuMinA(A_array);

	printf("GPU 计算结果：\n|R| %f，最大值 %f，最小值 %f\n",*Amp_R,*A_max,*A_min);
	//verifySeccuss=verify(data_c1,data_c2,1);

	//if(verifySeccuss)
	//	printf("Verify Seccuss.\n");
	//else
	//	printf("Verify Error!\n");
	cudaFree(data_R);
	cudaFree(Amp_R);
	cudaFree(SumR);
	cudaFree(A_max);
	cudaFree(A_min);
	cudaFree(MaxTemp);
	cudaFree(MinTemp);
	
	getch();
}