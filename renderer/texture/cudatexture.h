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

#include "buffer\buffers.h"

namespace ExposureRender
{

/*! Cuda texture template base class */
template<class T, int NoDimensions = 1>
class EXPOSURE_RENDER_DLL CudaTexture
{
public:
	/*! Constructor
		@param[in] Normalized Normalized element access
		@param[in] FilterMode Type of filtering
		@param[in] AddressMode Type of addressing near edges
	*/
	HOST CudaTexture(const bool& Normalized = true, const Enums::FilterMode& FilterMode = Enums::Linear, const Enums::AddressMode& AddressMode = Enums::Clamp) :
		Resolution(),
		Normalized(Normalized),
		FilterMode(FilterMode),
		AddressMode(AddressMode),
		TextureObject()
	{
	}
	
	/*! Destructor */
	HOST virtual ~CudaTexture(void)
	{
		this->Free();
		cudaDestroyTextureObject(this->TextureObject);
	}
	
	/*! Assignment operator
		@param[in] Other Cuda texture to copy from
		@return Copied cuda texture by reference
	*/
	HOST CudaTexture& operator = (const CudaTexture& Other)
	{
		throw (Exception(Enums::Error, "Not implemented yet!"));
	}
	
	/*! Free dynamic data owned by the texture */
	HOST void Free(void)
	{
	}
	
	/*! Gets the resolution
		@return Resolution
	*/
	HOST_DEVICE Vec<int, NoDimensions> GetResolution() const
	{
		return this->Resolution;
	}

protected:
	Vec<int, NoDimensions>	Resolution;			/*! Texture resolution */
	bool					Normalized;			/*! Whether texture access is in normalized texture coordinates */
	Enums::FilterMode		FilterMode;			/*! Type of filtering  */
	Enums::AddressMode		AddressMode;		/*! Type of addressing  */
	cudaTextureObject_t		TextureObject;		/*! Cuda texture object */
};

}
