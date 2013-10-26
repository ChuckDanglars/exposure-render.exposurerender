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

/*! \class ColorRGBuc
 * \brief RGB unsigned char class
 */
class EXPOSURE_RENDER_DLL ColorRGBuc : public Vec<unsigned char, 3>
{
public:
	/*! Default constructor */
	HOST_DEVICE ColorRGBuc()
	{
		for (int i = 0; i < 3; ++i)
			this->D[i] = 0;
	}

	/*! Construct and initialize with default value
		* \param V The default value
	*/
	HOST_DEVICE ColorRGBuc(const unsigned char& V)
	{
		for (int i = 0; i < 3; ++i)
			this->D[i] = V;
	}

	/*! Constructor with initializing values */
	HOST_DEVICE ColorRGBuc(const unsigned char& R, const unsigned char& G, const unsigned char& B)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
	}

	/*! Copy constructor */
	HOST_DEVICE ColorRGBuc(const Vec<unsigned char, 3>& Other)
	{
		for (int i = 0; i < 3; ++i)
			this->D[i] = Other[i];
	}
	
	/*! Constructs a black RGBAuc color */
	static HOST_DEVICE ColorRGBuc Black()
	{
		return ColorRGBuc();
	}
	
	/*! Test whether the color is black
		@return Black
	*/
	HOST_DEVICE bool IsBlack()
	{
		for (int i = 0; i < 3; i++)
			if (this->D[i] != 0)
				return false;
												
		return true;
	}
	
	/*! Determine the luminance
		@return Luminance
	*/
	HOST_DEVICE float Luminance() const
	{
		return 0.3f * D[0] + 0.59f * D[1]+ 0.11f * D[2];
	}
};

/*! Multiply ColorRGBuc with float
	* \param C ColorRGBuc
	* \param F Float to multiply with
	@return C x F
*/
static inline HOST_DEVICE ColorRGBuc operator * (const ColorRGBuc& C, const float& F)
{
	return ColorRGBuc(	(unsigned char)((float)C[0] * F),
						(unsigned char)((float)C[1] * F),
						(unsigned char)((float)C[2] * F));
};

/*! Multiply float with ColorRGBuc
	* \param C ColorRGBuc
	* \param F Float to multiply with
	@return F x C
*/
static inline HOST_DEVICE ColorRGBuc operator * (const float& F, const ColorRGBuc& C)
{
	return ColorRGBuc(	(unsigned char)((float)C[0] * F),
						(unsigned char)((float)C[1] * F),
						(unsigned char)((float)C[2] * F));
};

/*! Multiply two ColorRGBuc vectors
	* \param A Vector A
	* \param B Vector B
	@return A x B
*/
static inline HOST_DEVICE ColorRGBuc operator * (const ColorRGBuc& A, const ColorRGBuc& B)
{
	return ColorRGBuc(A[0] * B[0], A[1] * B[1], A[2] * B[2]);
};

/*! Divide ColorRGBuc vector by float value
	* \param C ColorRGBuc to divide
	* \param F Float to divide with
	@return F / V
*/
static inline HOST_DEVICE ColorRGBuc operator / (const ColorRGBuc& C, const float& F)
{
	// Compute F reciprocal, slightly faster
	const float InvF = (F == 0.0f) ? 0.0f : 1.0f / F;

	return ColorRGBuc((unsigned char)((float)C[0] * InvF), (unsigned char)((float)C[1] * InvF), (unsigned char)((float)C[2] * InvF));
};

/*! Subtract two ColorRGBuc vectors
	* \param A Vector A
	* \param B Vector B
	@return A - B
*/
static inline HOST_DEVICE ColorRGBuc operator - (const ColorRGBuc& A, const ColorRGBuc& B)
{
	return ColorRGBuc(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
};

/*! Add two ColorRGBuc vectors
	* \param A Vector A
	* \param B Vector B
	@return A + B
*/
static inline HOST_DEVICE ColorRGBuc operator + (const ColorRGBuc& A, const ColorRGBuc& B)
{
	return ColorRGBuc(A[0] + B[0], A[1] + B[1], A[2] + B[2]);
};

/*! Linearly interpolate two ColorRGBuc vectors
	* \param LerpC Interpolation coefficient
	* \param A Vector A
	* \param B Vector B
	@return Interpolated vector
*/
HOST_DEVICE inline ColorRGBuc Lerp(const float& LerpC, const ColorRGBuc& A, const ColorRGBuc& B)
{
	return (1.0f - LerpC) * A + LerpC * B;
}

/*! Computes the normalized color distance between \a A and \a B
	@param A Color A
	@param B Color B
	@return Normalized color distance
*/
HOST_DEVICE inline float NormalizedColorDistance(const ColorRGBuc& A, const ColorRGBuc& B)
{
	return ONE_OVER_255 * sqrtf(powf((float)(A[0] - B[0]), 2.0f) + powf((float)(A[1] - B[1]), 2.0f) + powf((float)(A[2] - B[2]), 2.0f));
}

}
