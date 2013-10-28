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

#include "texture\cudatexture.h"
#include "core\cudawrapper.h"

namespace ExposureRender
{

/*! 3D Cuda texture class */
class EXPOSURE_RENDER_DLL CudaTexture3D
{
public:
	/*! Constructor
		@param[in] Normalized Normalized element access
		@param[in] FilterMode Type of filtering
		@param[in] AddressMode Type of addressing near edges
	*/
	HOST CudaTexture3D() :
		Resolution(),
		Array(0),
		TextureObject()
	{
	}

	/*! Destructor */
	HOST virtual ~CudaTexture3D(void)
	{
		this->Free();

		cudaDestroyTextureObject(this->TextureObject);
	}

	/*! Free dynamic data owned by the texture */
	HOST void Free(void)
	{
		Cuda::FreeArray(this->Array);
	}

	/*! Get buffer element at position \a XY
		@param[in] XY XY position in buffer
		@return Interpolated value at \a XY
	*/
	DEVICE short operator()(const Vec3f& NormalizedUVW)
	{
#ifdef __CUDACC__
		return tex3D<float>(this->TextureObject, NormalizedUVW[0], NormalizedUVW[1], NormalizedUVW[2]) * (float)SHRT_MAX;
#else
		return 1000;
#endif
	}

	HOST void Create(const Vec<int, 3>& Resolution, short* HostData)
	{
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution = Resolution;

		cudaExtent CudaExtent;

		CudaExtent.width	= Resolution[0];
		CudaExtent.height	= Resolution[1];
		CudaExtent.depth	= Resolution[2];
	
		cudaChannelFormatDesc CudaChannelFormat = cudaCreateChannelDesc<short>();

		Cuda::HandleCudaError(cudaMalloc3DArray(&this->Array, &CudaChannelFormat, CudaExtent));

		if (this->Resolution.CumulativeProduct() == 0)
			return;

		cudaMemcpy3DParms CopyParams = { 0 };

		CopyParams.srcPtr	= make_cudaPitchedPtr(HostData, CudaExtent.width * sizeof(short), CudaExtent.width, CudaExtent.height);
		CopyParams.dstArray	= this->Array;
		CopyParams.extent	= CudaExtent;
		CopyParams.kind		= cudaMemcpyHostToDevice;
	
		Cuda::HandleCudaError(cudaMemcpy3D(&CopyParams));

		//Cuda::HandleCudaError(cudaMemcpyToArray(this->Array, 0, 0, HostData, this->Resolution.CumulativeProduct() * sizeof(short), cudaMemcpyHostToDevice));

		cudaResourceDesc CudaResource;
		
		memset(&CudaResource, 0, sizeof(CudaResource));

		CudaResource.resType			= cudaResourceTypeArray;
		CudaResource.res.array.array	= this->Array;

		cudaTextureDesc CudaTexture;
		memset(&CudaTexture, 0, sizeof(CudaTexture));

		CudaTexture.addressMode[0]		= cudaAddressModeWrap;
		CudaTexture.addressMode[1]		= cudaAddressModeWrap;
		CudaTexture.addressMode[2]		= cudaAddressModeWrap;
		CudaTexture.filterMode			= cudaFilterModeLinear;
		CudaTexture.readMode			= cudaReadModeNormalizedFloat;
		CudaTexture.normalizedCoords	= 1;

		Cuda::HandleCudaError(cudaCreateTextureObject(&this->TextureObject, &CudaResource, &CudaTexture, 0));
	}

private:
	Vec<int, 3>				Resolution;			/*! Texture resolution */
	cudaArray*				Array;				/*! Cuda array, in case of pitched texture memory */
	cudaTextureObject_t		TextureObject;		/*! Cuda texture object */
};

}
