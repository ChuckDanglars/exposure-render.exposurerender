
#include "tonemap.cuh"
#include "core\cudawrapper.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlToneMap(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	Film& Film = Renderer->Camera.GetFilm();

	if (X >= Film.GetWidth() || Y >= Film.GetHeight())
		return;

	CudaBuffer2D<ColorXYZAf>& IterationEstimateHDR 		= Film.GetIterationEstimateHDR();
	CudaBuffer2D<ColorRGBAuc>& IterationEstimateLDR		= Film.GetIterationEstimateLDR();

	ColorXYZAf RunningEstimateXYZ = IterationEstimateHDR(X, Y);

	RunningEstimateXYZ.ToneMap(0.1f);
	
	IterationEstimateLDR.Set(X, Y, ColorRGBAuc::FromXYZAf(RunningEstimateXYZ.D));
}

void ToneMap(Renderer* HostRenderer, Renderer* DevRenderer)
{
	LAUNCH_DIMENSIONS

	KrnlToneMap<<<Grid, Block>>>(DevRenderer);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Tone map");
}

}
