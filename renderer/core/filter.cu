
#include "filter.cuh"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlFilter(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
		return;

	int PID = Y * Camera->GetFilm().GetWidth() + X;
}




KERNEL void KrnlGaussianFilterHorizontalRGBAuc(int Radius, Buffer2D<ColorRGBAuc>* Input, Buffer2D<ColorRGBAuc>* Output)
{
	KERNEL_2D(Input->GetResolution()[0], Input->GetResolution()[1])
		
	const int Range[2] = 
	{
		max((int)ceilf(IDx - Radius), 0),
		min((int)floorf(IDx + Radius), Input->GetResolution()[0] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int x = Range[0]; x <= Range[1]; x++)
	{
		const float Weight = gpTracer->GaussianFilterTables.Weight(Radius, Radius + (IDx - x), Radius);

		Sum[0]		+= Weight * (*Input)(x, IDy)[0];
		Sum[1]		+= Weight * (*Input)(x, IDy)[1];
		Sum[2]		+= Weight * (*Input)(x, IDy)[2];
		Sum[3]		+= Weight * (*Input)(x, IDy)[3];
		SumWeight	+= Weight;
	}
	
	if (SumWeight > 0.0f)
	{
		(*Output)(IDx, IDy)[0] = Sum[0] / SumWeight;
		(*Output)(IDx, IDy)[1] = Sum[1] / SumWeight;
		(*Output)(IDx, IDy)[2] = Sum[2] / SumWeight;
		(*Output)(IDx, IDy)[3] = Sum[3] / SumWeight;
	}
	else
		(*Output)(IDx, IDy) = (*Input)(IDx, IDy);
}

KERNEL void KrnlGaussianFilterVerticalRGBAuc(int Radius, Buffer2D<ColorRGBAuc>* Input, Buffer2D<ColorRGBAuc>* Output)
{
	KERNEL_2D(Input->GetResolution()[0], Input->GetResolution()[1])
		
	const int Range[2] =
	{
		max((int)ceilf(IDy - Radius), 0),
		min((int)floorf(IDy + Radius), Input->GetResolution()[1] - 1)
	};

	ColorRGBAf Sum;

	float SumWeight = 0.0f;

	for (int y = Range[0]; y <= Range[1]; y++)
	{
		const float Weight = gpTracer->GaussianFilterTables.Weight(Radius, Radius, Radius + (IDy - y));

		Sum[0]		+= Weight * (*Input)(IDx, y)[0];
		Sum[1]		+= Weight * (*Input)(IDx, y)[1];
		Sum[2]		+= Weight * (*Input)(IDx, y)[2];
		Sum[3]		+= Weight * (*Input)(IDx, y)[3];
		SumWeight	+= Weight;
	}
	
	if (SumWeight > 0.0f)
	{
		(*Output)(IDx, IDy)[0] = Sum[0] / SumWeight;
		(*Output)(IDx, IDy)[1] = Sum[1] / SumWeight;
		(*Output)(IDx, IDy)[2] = Sum[2] / SumWeight;
		(*Output)(IDx, IDy)[3] = Sum[3] / SumWeight;
	}
	else
		(*Output)(IDx, IDy) = (*Input)(IDx, IDy);
}










void Filter(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlFilter<<<KernelGrid, KernelBlock>>>(DevCamera);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Filter");

	Cuda::HandleCudaError(cudaFree(DevCamera));
}

}
