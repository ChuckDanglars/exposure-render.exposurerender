
texture<unsigned short, 3, cudaReadModeNormalizedFloat> TexVolume;

#include "estimate.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"
#include "shapes\box.h"

namespace ExposureRender
{

KERNEL void KrnlEstimate(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Renderer->Camera.GetFilm().GetWidth() || Y >= Renderer->Camera.GetFilm().GetHeight())
		return;

	/*
	Halton1D H(2, blockIdx.x);

	float Value[3];

	halton_3(10000 + PID, Value);

	// float* Value = H.GetNext();

	Output[PID * 3 + 0] = Value[0] * 255.0f;
	Output[PID * 3 + 1] = Value[1] * 255.0f;
	Output[PID * 3 + 2] = Value[2] * 255.0f;
	*/
	
	RNG& Random = Renderer->Camera.GetFilm().GetRandomNumberGenerator(Vec2i(X, Y));

	Box B(Vec3f(0.1f));

	Ray R;
	
	Renderer->Camera.Sample(R, Vec2i(X, Y), Random);

	float T[2] = { 0.0f };

	bool Intersects = B.Intersect(R, T[0], T[1]);

	Renderer->Camera.GetFilm().GetIterationEstimateHDR().Set(X, Y, ColorXYZAf(Intersects ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));

	/*
	Output[PID * 3 + 0] = Intersects ? 255 : 0;
	Output[PID * 3 + 1] = 0;
	Output[PID * 3 + 2] = 0;
	*/
}

void Estimate(Renderer* HostRenderer, Renderer* DevRenderer)
{
	KrnlEstimate<<<HostRenderer->Camera.GetFilm().Grid, HostRenderer->Camera.GetFilm().Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Estimate");
}

}
