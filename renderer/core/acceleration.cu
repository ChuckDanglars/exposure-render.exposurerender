
#include "acceleration.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlAcceleration(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Renderer->Camera.GetFilm().GetWidth() || Y >= Renderer->Camera.GetFilm().GetHeight())
		return;
}

void Acceleration(Renderer* HostRenderer, Renderer* DevRenderer)
{
	KrnlAcceleration<<<HostRenderer->Camera.GetFilm().Grid, HostRenderer->Camera.GetFilm().Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Acceleration");
}

}
