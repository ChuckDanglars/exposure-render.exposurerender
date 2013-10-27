
#include "estimate.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlEstimate(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Renderer->Camera.GetFilm().GetWidth() || Y >= Renderer->Camera.GetFilm().GetHeight())
		return;
	
	CudaBuffer2D<ColorXYZAf>& IterationEstimateHDR = Renderer->Camera.GetFilm().GetIterationEstimateHDR();

	RNG Random = Renderer->Camera.GetFilm().GetRandomNumberGenerator(Vec2i(X, Y));

	Ray R;

	Renderer->Camera.Sample(R, Vec2i(X, Y), Random);

	ScatterEvent SE;

	if (Renderer->Volume.Intersect(R, Random, SE))
		IterationEstimateHDR.Set(X, Y, ColorXYZAf(1.0f, 1.0f, 1.0f, 0.0f));
	else
		IterationEstimateHDR.Set(X, Y, ColorXYZAf(0.0f, 0.0f, 0.0f, 0.0f));
	/*

	

	float T[2] = { 0.0f };

	bool Intersects = B.Intersect(R, T[0], T[1]);

	Renderer->Camera.GetFilm().GetIterationEstimateHDR().Set(X, Y, ColorXYZAf(Intersects ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));

	
	Output[PID * 3 + 0] = Intersects ? 255 : 0;
	Output[PID * 3 + 1] = 0;
	Output[PID * 3 + 2] = 0;
	*/
}

void Estimate(Renderer* HostRenderer, Renderer* DevRenderer)
{
	LAUNCH_DIMENSIONS

	KrnlEstimate<<<Grid, Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Estimate");
}

}
