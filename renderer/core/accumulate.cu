
#include "accumulate.cuh"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlAccumulate(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;

	CudaBuffer2D<ColorRGBAuc>& Estimate		= Camera->GetFilm().GetIterationEstimateLDR();
	CudaBuffer2D<ColorRGBAul>& Accumulate	= Camera->GetFilm().GetAccumulatedEstimate();

	Accumulate(X, Y)[0] += Estimate(X, Y)[0];
	Accumulate(X, Y)[1] += Estimate(X, Y)[1];
	Accumulate(X, Y)[2] += Estimate(X, Y)[2];
	Accumulate(X, Y)[3] += Estimate(X, Y)[3];
}

void Accumulate(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlAccumulate<<<KernelGrid, KernelBlock>>>(DevCamera);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Accumulate");

	Cuda::HandleCudaError(cudaFree(DevCamera));
}

}