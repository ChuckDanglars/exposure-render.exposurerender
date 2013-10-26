
#include "integrate.cuh"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlIntegrate(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;

	CudaBuffer2D<ColorRGBAul>& AccumulatedEstimate	= Camera->GetFilm().GetAccumulatedEstimate();
	CudaBuffer2D<ColorRGBuc>& CudaRunningEstimate	= Camera->GetFilm().GetCudaRunningEstimate();

	for (int c = 0; c < 3; c++)
		CudaRunningEstimate(X, Y)[c] = (unsigned char)((float)AccumulatedEstimate(X, Y)[c] / (float)Camera->GetFilm().GetNoEstimates());
}

void Integrate(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlIntegrate<<<KernelGrid, KernelBlock>>>(DevCamera);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Integrate");

	Cuda::HandleCudaError(cudaFree(DevCamera));

	Cuda::HandleCudaError(cudaMemcpy(HostCamera.GetFilm().GetHostRunningEstimate().GetData(), HostCamera.GetFilm().GetCudaRunningEstimate().GetData(), HostCamera.GetFilm().GetCudaRunningEstimate().GetNoBytes(), cudaMemcpyDeviceToHost));
}

}
