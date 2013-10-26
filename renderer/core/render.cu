
#include "render.cuh"
#include "geometry\lds.h"
#include "geometry\ray.h"
#include "vector\vector.h"
#include "geometry\box.h"
#include "core\renderer.h"
#include "core\cudawrapper.h"

namespace ExposureRender
{

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlRender(Camera* C, int Width, int Height, unsigned char* Output)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= Width || Y >= Height)
		return;

	int PID = Y * Width + X;
	/*
	Halton1D H(2, blockIdx.x);

	float Value[3];

	halton_3(10000 + PID, Value);

	// float* Value = H.GetNext();

	Output[PID * 3 + 0] = Value[0] * 255.0f;
	Output[PID * 3 + 1] = Value[1] * 255.0f;
	Output[PID * 3 + 2] = Value[2] * 255.0f;
	*/
	
	Box B(Vec3f(0.1f));

	Ray R;
	
	C->Sample(R, Vec2i(X, Y));

	float T[2] = { 0.0f };

	bool Intersects = B.Intersect(R, T[0], T[1]);

	Output[PID * 3 + 0] = Intersects ? 255 : 0;
	Output[PID * 3 + 1] = 0;
	Output[PID * 3 + 2] = 255;
}

void Render(float Position[3], float FocalPoint[3], float ViewUp[3], const int& Width, const int& Height, unsigned char* Output)
{
	Camera Cam;

	Cam.SetPos(Vec3f(Position));
	Cam.SetTarget(Vec3f(FocalPoint));
	Cam.SetUp(Vec3f(ViewUp));

	Cam.Update();

	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	const long ImageSize = Width * Height * 3 * sizeof(unsigned char);

	Camera* DevCamera = 0;
	unsigned char* DevOutput = 0;

	Cuda::HandleCudaError(cudaMalloc((void**)&DevOutput, ImageSize));
	Cuda::HandleCudaError(cudaMalloc((void**)&DevCamera, sizeof(Camera)));

	Cuda::HandleCudaError(cudaMemcpy(DevCamera, &Cam, sizeof(Camera), cudaMemcpyHostToDevice));

	KrnlRender<<<KernelGrid, KernelBlock>>>(DevCamera, Width, Height, DevOutput);
	cudaThreadSynchronize();
	Cuda::HandleCudaError(cudaGetLastError(), "Estimate");

	Cuda::HandleCudaError(cudaMemcpy(Output, DevOutput, ImageSize, cudaMemcpyDeviceToHost));

	Cuda::HandleCudaError(cudaFree(DevCamera));
	Cuda::HandleCudaError(cudaFree(DevOutput));
}

}
