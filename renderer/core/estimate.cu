
#include "estimate.cuh"
#include "core\camera.h"
#include "core\cudawrapper.h"
#include "shapes\box.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlEstimate(Camera* Camera)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Camera->GetFilm().GetWidth() || Y >= Camera->GetFilm().GetHeight())
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
	
	RNG& Random = Camera->GetFilm().GetRandomNumberGenerator(Vec2i(X, Y));

	Box B(Vec3f(0.1f));

	Ray R;
	
	Camera->Sample(R, Vec2i(X, Y), Random);

	float T[2] = { 0.0f };

	bool Intersects = B.Intersect(R, T[0], T[1]);

	Camera->GetFilm().GetIterationEstimateHDR().Set(X, Y, ColorXYZAf(Intersects ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));

	/*
	Output[PID * 3 + 0] = Intersects ? 255 : 0;
	Output[PID * 3 + 1] = 0;
	Output[PID * 3 + 2] = 0;
	*/
}

void Estimate(Camera& HostCamera)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)HostCamera.GetFilm().GetWidth() / (float)KernelBlock.x), (int)ceilf((float)HostCamera.GetFilm().GetHeight() / (float)KernelBlock.y));

	Camera* DevCamera = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &HostCamera, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(DevCamera);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Estimate");

	Cuda::HandleCudaError(cudaFree(DevCamera));
}

}
