
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

#define KRNL_BLOCK_W		8
#define KRNL_BLOCK_H		8
#define KRNL_BLOCK_SIZE		KRNL_BLOCK_W * KRNL_BLOCK_H

void Render(Renderer* HostRenderer)
{
	Film& Film = HostRenderer->Camera.GetFilm();

	if (Film.GetNoEstimates() == 1)
	{
		Film.GetAccumulatedEstimate().Reset();

		Film.GetRandomSeeds1().FromHost(Film.GetHostRandomSeeds1().GetData());
		Film.GetRandomSeeds2().FromHost(Film.GetHostRandomSeeds2().GetData());
	}

	const dim3 Block(KRNL_BLOCK_W, KRNL_BLOCK_H);
	const dim3 Grid((int)ceilf((float)Film.GetWidth() / (float)Block.x), (int)ceilf((float)Film.GetHeight() / (float)Block.y));

	Renderer* DevRenderer = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevRenderer, sizeof(Renderer)));

	Cuda::HandleCudaError(cudaMemcpy(DevRenderer, HostRenderer, sizeof(Renderer), cudaMemcpyHostToDevice));

	Estimate(Grid, Block, HostRenderer, DevRenderer);
	ToneMap(Grid, Block, HostRenderer, DevRenderer);
	Filter(Grid, Block, HostRenderer, DevRenderer);
	Accumulate(Grid, Block, HostRenderer, DevRenderer);
	Integrate(Grid, Block, HostRenderer, DevRenderer);

	Cuda::HandleCudaError(cudaFree(DevRenderer));
}

}
