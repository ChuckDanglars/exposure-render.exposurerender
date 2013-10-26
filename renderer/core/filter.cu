
#include "filter.cuh"
#include "core\cudawrapper.h"
#include "core\utilities.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlGaussianFilterHorizontal(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;
	
	CudaBuffer2D<ColorRGBAuc>& Input	= Camera->GetFilm().GetIterationEstimateLDR();
	CudaBuffer2D<ColorRGBAuc>& Output	= Camera->GetFilm().GetIterationEstimateTempFilterLDR();

	const int Range[2] = 
	{
		Max((int)ceilf(X - 1), 0),
		Min((int)floorf(X + 1), Input.GetResolution()[0] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int x = Range[0]; x <= Range[1]; x++)
	{
		const float Weight = Camera->GetFilm().GetGaussianFilterWeights()[x - X];

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

KERNEL void KrnlGaussianFilterVertical(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;
	
	CudaBuffer2D<ColorRGBAuc>& Input	= Camera->GetFilm().GetIterationEstimateTempFilterLDR();
	CudaBuffer2D<ColorRGBAuc>& Output	= Camera->GetFilm().GetIterationEstimateLDR();

	const int Range[2] =
	{
		Max((int)ceilf(Y - 1), 0),
		Min((int)floorf(Y + 1), Input.GetResolution()[1] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int y = Range[0]; y <= Range[1]; y++)
	{
		const float Weight = Camera->GetFilm().GetGaussianFilterWeights()[y - Y];

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


void Filter(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlGaussianFilterHorizontal<<<KernelGrid, KernelBlock>>>(DevCamera);
	KrnlGaussianFilterVertical<<<KernelGrid, KernelBlock>>>(DevCamera);

	cudaThreadSynchronize();

	Cuda::HandleCudaError(cudaGetLastError(), "Filter");

	Cuda::HandleCudaError(cudaFree(DevCamera));
}

}
