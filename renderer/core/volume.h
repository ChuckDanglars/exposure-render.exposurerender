/*
*	@file
*	@author  Thomas Kroes <t.kroes at tudelft.nl>
*	@version 1.0
*	
*	@section LICENSE
*	
*	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
*	
*	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
*	Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
*
*	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "geometry\boundingbox.h"
#include "geometry\transform.h"
#include "geometry\scatterevent.h"
#include "core\utilities.h"
#include "core\rng.h"
#include "core\tracer.h"

namespace ExposureRender
{
			
/*! Volume class */
class EXPOSURE_RENDER_DLL Volume
{
public:
	/*! Default constructor */
	HOST Volume() :
		Transform(),
		BoundingBox(),
		Resolution(0),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Size(1.0f),
		InvSize(1.0f),
		Array(0),
		TextureObject(),
		AcceleratorType(Enums::Octree),
		Tracer()
	{
	}

	/*! Destructor */
	HOST virtual ~Volume(void)
	{
		Cuda::HandleCudaError(cudaFree(this->Array));
		Cuda::HandleCudaError(cudaDestroyTextureObject(this->TextureObject));
	}

	/*! Computes the gradient at \a P using central differences
		@param[in] P Position at which to compute the gradient
		@return Gradient at \a P
	*/
	HOST void Create(const Vec3i& Resolution, const Vec3f& Spacing, short* Voxels, const Matrix44& Matrix = Matrix44())
	{
		this->Resolution = Resolution;

		this->Transform.Set(Matrix);

		this->Spacing		= Spacing;
		this->InvSpacing	= 1.0f / this->Spacing;
		this->Size			= Vec3f(Resolution[0] * this->Spacing[0], Resolution[1] * this->Spacing[1], Resolution[2] * this->Spacing[2]);
		this->InvSize		= 1.0f / this->Size;

		this->BoundingBox.SetMinP(Vec3f(0.0f, 0.0f, 0.0f));
		this->BoundingBox.SetMaxP(this->Size);

		Cuda::HandleCudaError(cudaFree(this->Array));

		this->Array = 0;

		cudaExtent CudaExtent;

		CudaExtent.width	= this->Resolution[0];
		CudaExtent.height	= this->Resolution[1];
		CudaExtent.depth	= this->Resolution[2];
	
		cudaChannelFormatDesc CudaChannelFormat = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);

		Cuda::HandleCudaError(cudaMalloc3DArray(&this->Array, &CudaChannelFormat, CudaExtent));

		if (this->Resolution.CumulativeProduct() == 0)
			return;

		cudaMemcpy3DParms CopyParams = { 0 };

		CopyParams.srcPtr	= make_cudaPitchedPtr(Voxels, CudaExtent.width * sizeof(short), CudaExtent.width, CudaExtent.height);
		CopyParams.dstArray	= this->Array;
		CopyParams.extent	= CudaExtent;
		CopyParams.kind		= cudaMemcpyHostToDevice;
	
		Cuda::HandleCudaError(cudaMemcpy3D(&CopyParams));

		cudaResourceDesc CudaResource;
		
		memset(&CudaResource, 0, sizeof(CudaResource));

		CudaResource.resType			= cudaResourceTypeArray;
		CudaResource.res.array.array	= this->Array;

		cudaTextureDesc CudaTexture;
		memset(&CudaTexture, 0, sizeof(CudaTexture));

		CudaTexture.addressMode[0]		= cudaAddressModeWrap;
		CudaTexture.addressMode[1]		= cudaAddressModeWrap;
		CudaTexture.addressMode[2]		= cudaAddressModeWrap;
		CudaTexture.filterMode			= cudaFilterModePoint;
		CudaTexture.readMode			= cudaReadModeElementType;
		CudaTexture.normalizedCoords	= 1;

		Cuda::HandleCudaError(cudaCreateTextureObject(&this->TextureObject, &CudaResource, &CudaTexture, 0));

		
	}
	
	/*! Gets the (interpolated) voxel value at \a P
		@param[in] P Position in volume coordinate space
		@return Voxel value at \a P
	*/
	DEVICE short GetIntensity(const Vec3f& P)
	{
#ifdef __CUDACC__
		return tex3D<short>(this->TextureObject, P[0] * this->InvSize[0], P[1]* this->InvSize[1], P[2]* this->InvSize[2]);// * (float)SHRT_MAX;
#else
		return 0;
#endif
	}

	

	

	GET_SET_MACRO(HOST_DEVICE, Transform, Transform)
	GET_MACRO(HOST_DEVICE, BoundingBox, BoundingBox)
	GET_MACRO(HOST_DEVICE, Resolution, Vec3i)
	GET_MACRO(HOST_DEVICE, Spacing, Vec3f)
	GET_MACRO(HOST_DEVICE, InvSpacing, Vec3f)
	GET_MACRO(HOST_DEVICE, Size, Vec3f)
	GET_MACRO(HOST_DEVICE, InvSize, Vec3f)
	GET_SET_MACRO(HOST_DEVICE, AcceleratorType, Enums::AcceleratorType)
	GET_REF_MACRO(HOST_DEVICE, Tracer, Tracer)

private:
	Vec3i						Resolution;			/*! Texture resolution */
	Transform					Transform;			/*! Transform of the volume */
	Vec3f						InvSize;			/*! Inverse volume size */
	Vec3f						Spacing;			/*! Voxel spacing */
	Vec3f						InvSpacing;			/*! Inverse voxel spacing */
	Vec3f						Size;				/*! Volume size */
	Enums::AcceleratorType		AcceleratorType;	/*! Type of ray traversal accelerator */
	Tracer						Tracer;				/*! Tracer */
	cudaArray*					Array;				/*! Cuda array, used by the texture object */
	cudaTextureObject_t			TextureObject;		/*! Cuda texture object */
	BoundingBox					BoundingBox;		/*! Encompassing bounding box */
};

}