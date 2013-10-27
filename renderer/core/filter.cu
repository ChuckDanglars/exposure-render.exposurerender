
#include "filter.cuh"
#include "core\cudawrapper.h"
#include "core\utilities.h"
#include "core\renderer.h"

namespace ExposureRender
{

KERNEL void KrnlGaussianFilterHorizontal(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	Film& Film = Renderer->Camera.GetFilm();

	if (X >= Film.GetWidth() || Y >= Film.GetHeight())
		return;
	
	CudaBuffer2D<ColorRGBAuc>& Input	= Film.GetIterationEstimateLDR();
	CudaBuffer2D<ColorRGBAuc>& Output	= Film.GetIterationEstimateTempFilterLDR();

	const int Range[2] = 
	{
		Max((int)ceilf(X - 1), 0),
		Min((int)floorf(X + 1), Input.GetResolution()[0] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int x = Range[0]; x <= Range[1]; x++)
	{
		const float Weight = Film.GetGaussianFilterWeights()[x - X];

		Sum[0]		+= Weight * Input(x, Y)[0];
		Sum[1]		+= Weight * Input(x, Y)[1];
		Sum[2]		+= Weight * Input(x, Y)[2];
		Sum[3]		+= Weight * Input(x, Y)[3];
		SumWeight	+= Weight;
	}

	if (SumWeight > 0.0f)
	{
		Output(X, Y)[0] = Sum[0] / SumWeight;
		Output(X, Y)[1] = Sum[1] / SumWeight;
		Output(X, Y)[2] = Sum[2] / SumWeight;
		Output(X, Y)[3] = Sum[3] / SumWeight;
	}
	else
		Output(X, Y) = Input(X, Y);
}

KERNEL void KrnlGaussianFilterVertical(Renderer* Renderer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	Film& Film = Renderer->Camera.GetFilm();

	if (X >= Film.GetWidth() || Y >= Film.GetHeight())
		return;
	
	CudaBuffer2D<ColorRGBAuc>& Input	= Film.GetIterationEstimateTempFilterLDR();
	CudaBuffer2D<ColorRGBAuc>& Output	= Film.GetIterationEstimateLDR();

	const int Range[2] =
	{
		Max((int)ceilf(Y - 1), 0),
		Min((int)floorf(Y + 1), Input.GetResolution()[1] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int y = Range[0]; y <= Range[1]; y++)
	{
		const float Weight = Film.GetGaussianFilterWeights()[y - Y];

		Sum[0]		+= Weight * Input(X, y)[0];
		Sum[1]		+= Weight * Input(X, y)[1];
		Sum[2]		+= Weight * Input(X, y)[2];
		Sum[3]		+= Weight * Input(X, y)[3];
		SumWeight	+= Weight;
	}
	
	if (SumWeight > 0.0f)
	{
		Output(X, Y)[0] = Sum[0] / SumWeight;
		Output(X, Y)[1] = Sum[1] / SumWeight;
		Output(X, Y)[2] = Sum[2] / SumWeight;
		Output(X, Y)[3] = Sum[3] / SumWeight;
	}
	else
		Output(X, Y) = Input(X, Y);
}

void Filter(Renderer* HostRenderer, Renderer* DevRenderer)
{
	LAUNCH_DIMENSIONS

	KrnlGaussianFilterHorizontal<<<Grid, Block>>>(DevRenderer);
	KrnlGaussianFilterVertical<<<Grid, Block>>>(DevRenderer);

	cudaThreadSynchronize();

	Cuda::HandleCudaError(cudaGetLastError(), "Filter");
}

}
