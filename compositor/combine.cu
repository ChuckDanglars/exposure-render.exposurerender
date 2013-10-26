
#include "combine.cuh"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

struct Images
{
	unsigned char* Data[20];
	int NoImages;
};

KERNEL void KrnlCombine(int Width, int Height, Images* Estimates, unsigned char* Estimate)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Width || Y >= Height)
		return;

	int PID = Y * Width + X;

	int Sum[3] = { 0, 0, 0 };

	for (int e = 0; e < Estimates->NoImages; e++)
	{
		for (int c = 0; c < 3; c++)
			Sum[c] += Estimates->Data[e][PID * 3 + c];
	}

	const float InvNoEstimates = 1.0f / (float)Estimates->NoImages;
	
	for (int c = 0; c < 3; c++)
		Estimate[PID * 3 + c] = (unsigned char)(InvNoEstimates * (float)Sum[c]);
}

float Combine(int Width, int Height, unsigned char* Estimates[20], const int& NoEstimates, unsigned char* Estimate)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	cudaEvent_t Start;
	cudaEvent_t Stop;

	Cuda::HandleCudaError(cudaEventCreate(&Start));
	Cuda::HandleCudaError(cudaEventCreate(&Stop));
	Cuda::HandleCudaError(cudaEventRecord(Start, 0));

	Images HostEstimates;

	HostEstimates.NoImages = NoEstimates;

	for (int e = 0; e < NoEstimates; e++)
	{
		Cuda::HandleCudaError(cudaMalloc((void**)&(HostEstimates.Data[e]), Width * Height * 3));
		Cuda::HandleCudaError(cudaMemcpy(HostEstimates.Data[e], Estimates[e], Width * Height * 3, cudaMemcpyHostToDevice));
	}

	Images* DevEstimates = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevEstimates, sizeof(Images)));
	Cuda::HandleCudaError(cudaMemcpy(DevEstimates, &HostEstimates, sizeof(Images), cudaMemcpyHostToDevice));

	unsigned char* DevEstimate = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevEstimate, Width * Height * 3));
	
	KrnlCombine<<<KernelGrid, KernelBlock>>>(Width, Height, DevEstimates, DevEstimate);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Estimate");

	Cuda::HandleCudaError(cudaMemcpy(Estimate, DevEstimate, Width * Height * 3, cudaMemcpyDeviceToHost));

	Cuda::HandleCudaError(cudaFree(DevEstimate));
	Cuda::HandleCudaError(cudaFree(DevEstimates));
	
	for (int e = 0; e < NoEstimates; e++)
		Cuda::HandleCudaError(cudaFree(HostEstimates.Data[e]));

	Cuda::HandleCudaError(cudaEventRecord(Stop, 0));
	Cuda::HandleCudaError(cudaEventSynchronize(Stop));

	float TimeDelta = 0.0f;

	Cuda::HandleCudaError(cudaEventElapsedTime(&TimeDelta, Start, Stop));
	Cuda::HandleCudaError(cudaEventDestroy(Start));
	Cuda::HandleCudaError(cudaEventDestroy(Stop));

	return TimeDelta;
}

}
