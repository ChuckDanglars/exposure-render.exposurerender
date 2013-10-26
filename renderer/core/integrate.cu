
#include "integrate.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlIntegrate(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	Film& Film = Renderer->Camera.GetFilm();

	if (X >= Film.GetWidth() || Y >= Film.GetHeight())
		return;

	CudaBuffer2D<ColorRGBAul>& AccumulatedEstimate	= Film.GetAccumulatedEstimate();
	CudaBuffer2D<ColorRGBuc>& CudaRunningEstimate	= Film.GetCudaRunningEstimate();

	for (int c = 0; c < 3; c++)
		CudaRunningEstimate(X, Y)[c] = (unsigned char)((float)AccumulatedEstimate(X, Y)[c] / (float)Film.GetNoEstimates());
}

void Integrate(dim3 Grid, dim3 Block, Renderer* HostRenderer, Renderer* DevRenderer)
{
	KrnlIntegrate<<<Grid, Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Integrate");

	Cuda::HandleCudaError(cudaFree(DevRenderer));

	Cuda::HandleCudaError(cudaMemcpy(HostRenderer->Camera.GetFilm().GetHostRunningEstimate().GetData(), HostRenderer->Camera.GetFilm().GetCudaRunningEstimate().GetData(), HostRenderer->Camera.GetFilm().GetCudaRunningEstimate().GetNoBytes(), cudaMemcpyDeviceToHost));
}

}
