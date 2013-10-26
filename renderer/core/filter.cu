
#include "filter.cuh"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlFilter(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;

	int PID = Y * Camera->GetFilm().GetWidth() + X;
}

void Filter(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlFilter<<<KernelGrid, KernelBlock>>>(DevCamera);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Filter");

	Cuda::HandleCudaError(cudaFree(DevCamera));
}

}
