
#include "render.cuh"
#include "core\estimate.cuh"
#include "core\tonemap.cuh"
#include "core\filter.cuh"
#include "core\integrate.cuh"
#include "core\accumulate.cuh"
#include "core\camera.h"
#include "core\renderer.h"

namespace ExposureRender
{

void Render(Renderer* HostRenderer)
{
	Film& Film = HostRenderer->Camera.GetFilm();

	if (Film.GetNoEstimates() == 1)
	{
		Film.GetAccumulatedEstimate().Reset();

		Film.GetRandomSeeds1().FromHost(Film.GetHostRandomSeeds1().GetData());
		Film.GetRandomSeeds2().FromHost(Film.GetHostRandomSeeds2().GetData());
	}

	Renderer* DevRenderer = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevRenderer, sizeof(Renderer)));
	Cuda::HandleCudaError(cudaMemcpy(DevRenderer, HostRenderer, sizeof(Renderer), cudaMemcpyHostToDevice));

	Estimate(HostRenderer, DevRenderer);
	ToneMap(HostRenderer, DevRenderer);
	Filter(HostRenderer, DevRenderer);
	Accumulate(HostRenderer, DevRenderer);
	Integrate(HostRenderer, DevRenderer);

	Cuda::HandleCudaError(cudaFree(DevRenderer));

	Film.IncrementNoEstimates();

	Cuda::HandleCudaError(cudaMemcpy(Film.GetHostRunningEstimate().GetData(), Film.GetCudaRunningEstimate().GetData(), Film.GetCudaRunningEstimate().GetNoBytes(), cudaMemcpyDeviceToHost));
}

}
