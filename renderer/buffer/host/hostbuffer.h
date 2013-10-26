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

#include "vector\vector.h"

namespace ExposureRender
{

/*! \class Hostbuffer
 * \brief Base host buffer class
 */
template<class T, int NoDimensions = 1>
class EXPOSURE_RENDER_DLL HostBuffer
{
public:
	/*! Constructor
		@param[in] FilterMode Type of filtering
		@param[in] AddressMode Type of addressing near edges
	*/
	HOST HostBuffer(Enums::FilterMode FilterMode = Enums::Linear, Enums::AddressMode AddressMode = Enums::Wrap) :
		FilterMode(FilterMode),
		AddressMode(AddressMode),
		Data(NULL),
		Resolution()
	{
	}
	
	/*! Copy constructor */
	HOST HostBuffer(const HostBuffer& Other)
	{
		*this = Other;
	}

	/*! Destructor */
	HOST virtual ~HostBuffer(void)
	{
		this->Free();
	}
	
	/*! Assignment operator
		@param[in] Other HostBuffer to copy from
		@return Copied buffer by reference
	*/
	HOST HostBuffer& operator = (const HostBuffer& Other)
	{
		this->FilterMode	= Other.FilterMode;
		this->AddressMode	= Other.AddressMode;
		
		this->Resize(Other.GetResolution());

		memcpy(this->Data, Other.Data, this->GetNoBytes());

		return *this;
	}
	
	/*! Frees the memory owned by the buffer */
	HOST void Free(void)
	{
		free(this->Data);
		this->Data = NULL;
		
		this->Resolution = Vec<int, NoDimensions>();
	}
	
	/*! Resets the memory owned by the buffer */
	HOST void Reset(void)
	{
		if (this->Resolution.CumulativeProduct() <= 0)
			return;

		memset(this->Data, 0, this->GetNoBytes());
	}
	
	/*! Resize the buffer
		@param[in] Resolution Resolution of the buffer
	*/
	HOST void Resize(const Vec<int, NoDimensions>& Resolution)
	{
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution = Resolution;

		if (this->Resolution.CumulativeProduct() <= 0)
			return;

		const int NoBytes = this->GetNoBytes();

		this->Data = (T*)malloc(NoBytes);

		memset(this->Data, 0, NoBytes);
		
		this->Reset();
	}

	/*! Get element at index \a ID
		@param[in] ID Index
		@return Element at \a ID
	*/
	HOST_DEVICE T& operator[](const int& ID) const
	{
		return this->Data[Clamp(ID, 0, this->GetNoElements() - 1)];
	}

	/*! Gets the number of bytes
		@return Number of bytes occupied by the buffer
	*/
	HOST_DEVICE virtual long GetNoBytes(void) const
	{
		return this->Resolution.CumulativeProduct() * sizeof(T);
	} 
	
	/*! Gets a pointer to the data
		@return Pointer to raw data
	*/
	HOST_DEVICE T* GetData() const
	{
		return this->Data;
	}

	/*! Gets the filter mode
		@return Filter mode
	*/
	HOST Enums::FilterMode GetFilterMode() const
	{
		return this->FilterMode;
	}

	/*! Sets the filter mode
		@param[in] FilterMode FilterMode
	*/
	HOST void SetFilterMode(const Enums::FilterMode& FilterMode)
	{
		this->FilterMode = FilterMode;
	}

	/*! Gets the address mode 
		@return Address mode
	*/
	HOST Enums::AddressMode GetAddressMode() const
	{
		return this->AddressMode;
	}

	/*! Gets the buffer's resolution 
		@return Resolution
	*/
	HOST_DEVICE Vec<int, NoDimensions> GetResolution() const
	{
		return this->Resolution;
	}

	/*! Gets the number of elements in the buffer 
		@return Number of elements
	*/
	HOST_DEVICE int GetNoElements() const
	{
		return this->Resolution.CumulativeProduct();
	}
	
protected:
	Enums::FilterMode			FilterMode;						/*! Type of filtering  */
	Enums::AddressMode			AddressMode;					/*! Type of addressing  */
	T*							Data;							/*! Pointer to raw data on host/device */
	Vec<int, NoDimensions>		Resolution;						/*! HostBuffer resolution */
};

}
