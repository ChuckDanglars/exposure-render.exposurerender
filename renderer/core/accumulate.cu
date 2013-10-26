
#include "accumulate.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlAccumulate(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Renderer->Camera.GetFilm().GetWidth() || Y >= Renderer->Camera.GetFilm().GetHeight())
		return;

	CudaBuffer2D<ColorRGBAuc>& Estimate		= Renderer->Camera.GetFilm().GetIterationEstimateLDR();
	CudaBuffer2D<ColorRGBAul>& Accumulate	= Renderer->Camera.GetFilm().GetAccumulatedEstimate();

	Accumulate(X, Y)[0] += Estimate(X, Y)[0];
	Accumulate(X, Y)[1] += Estimate(X, Y)[1];
	Accumulate(X, Y)[2] += Estimate(X, Y)[2];
	Accumulate(X, Y)[3] += Estimate(X, Y)[3];
}

void Accumulate(Renderer* HostRenderer, Renderer* DevRenderer)
{
	KrnlAccumulate<<<HostRenderer->Camera.GetFilm().Grid, HostRenderer->Camera.GetFilm().Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Accumulate");
}

}