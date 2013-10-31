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

#include "core\volume.h"
#include "transferfunction\transferfunctions.h"
#include "color\color.h"

namespace ExposureRender
{

/*! Computes the gradient in volume \a V at \a P using central differences
	@param[in] V Input volume
	@param[in] P Position in volume space at which to compute the gradient
	@return Gradient at \a P
*/
DEVICE Vec3f GradientCD(Volume& V, const Vec3f& P)
{
	const float Intensity[3][2] = 
	{
		{ V.GetIntensity(P + Vec3f(V.GetSpacing()[0], 0.0f, 0.0f)), V.GetIntensity(P - Vec3f(V.GetSpacing()[0], 0.0f, 0.0f)) },
		{ V.GetIntensity(P + Vec3f(0.0f, V.GetSpacing()[1], 0.0f)), V.GetIntensity(P - Vec3f(0.0f, V.GetSpacing()[1], 0.0f)) },
		{ V.GetIntensity(P + Vec3f(0.0f, 0.0f, V.GetSpacing()[2])), V.GetIntensity(P - Vec3f(0.0f, 0.0f, V.GetSpacing()[2])) }
	};

	return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}
	
/*! Computes the gradient in volume \a V at \a P using forward differences
	@param[in] V Input volume
	@param[in] P Position in volume space at which to compute the gradient
	@return Gradient at \a P
*/
DEVICE Vec3f GradientFD(Volume& V, const Vec3f& P)
{
	const float Intensity[4] = 
	{
		V.GetIntensity(P),
		V.GetIntensity(P + Vec3f(V.GetSpacing()[0], 0.0f, 0.0f)),
		V.GetIntensity(P + Vec3f(0.0f, V.GetSpacing()[1], 0.0f)),
		V.GetIntensity(P + Vec3f(0.0f, 0.0f, V.GetSpacing()[2]))
	};

	return Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
}
	
/*! Computes the filtered gradient in volume \a V at \a P using central differences
	@param[in] V Input volume
	@param[in] P Position in volume space at which to compute the gradient
	@return Gradient at \a P
*/
DEVICE Vec3f GradientFiltered(Volume& V, const Vec3f& P)
{
	Vec3f Offset(Vec3f(V.GetSpacing()[0], 0.0f, 0.0f)[0], Vec3f(0.0f, V.GetSpacing()[1], 0.0f)[1], Vec3f(0.0f, 0.0f, V.GetSpacing()[2])[2]);

	const Vec3f G0 = GradientCD(V, P);
	const Vec3f G1 = GradientCD(V, P + Vec3f(-Offset[0], -Offset[1], -Offset[2]));
	const Vec3f G2 = GradientCD(V, P + Vec3f( Offset[0],  Offset[1],  Offset[2]));
	const Vec3f G3 = GradientCD(V, P + Vec3f(-Offset[0],  Offset[1], -Offset[2]));
	const Vec3f G4 = GradientCD(V, P + Vec3f( Offset[0], -Offset[1],  Offset[2]));
	const Vec3f G5 = GradientCD(V, P + Vec3f(-Offset[0], -Offset[1],  Offset[2]));
	const Vec3f G6 = GradientCD(V, P + Vec3f( Offset[0],  Offset[1], -Offset[2]));
	const Vec3f G7 = GradientCD(V, P + Vec3f(-Offset[0],  Offset[1],  Offset[2]));
	const Vec3f G8 = GradientCD(V, P + Vec3f( Offset[0], -Offset[1], -Offset[2]));
	    
	const Vec3f L0 = Lerp(0.5f, Lerp(0.5f, G1, G2), Lerp(0.5f, G3, G4));
	const Vec3f L1 = Lerp(0.5f, Lerp(0.5f, G5, G6), Lerp(0.5f, G7, G8));
	    
	return Lerp(0.75f, G0, Lerp(0.5f, L0, L1));
}
	
/*! Computes the gradient in volume \a V at \a P using \a GradientMode
	@param[in] V Input volume
	@param[in] P Position in volume space at which to compute the gradient
	@param[in] GradientMode Type of gradient computation
	@return Gradient at \a P
*/
DEVICE Vec3f Gradient(Volume& V, const Vec3f& P, const Enums::GradientMode& GradientMode)
{
	switch (GradientMode)
	{
		case Enums::ForwardDifferences:		return GradientFD(V, P);
		case Enums::CentralDifferences:		return GradientCD(V, P);
		case Enums::Filtered:				return GradientFiltered(V, P);
	}

	return GradientFD(V, P);
}
	
/*! Computes the normalized gradient in volume \a V at \a P using \a GradientMode
	@param[in] V Input volume
	@param[in] P Position in volume space at which to compute the gradient
	@param[in] GradientMode Type of gradient computation
	@return Gradient at \a P
*/
DEVICE Vec3f NormalizedGradient(Volume& V, const Vec3f& P, const Enums::GradientMode& GradientMode)
{
	return Normalize(Gradient(V, P, GradientMode));
}
	
/*! Computes the gradient magnitude in volume \a V at \a P
	@param[in] V Input volume
	@param[in] P Position at which to compute the gradient magnitude
	@return Gradient magnitude at \a P
*/
DEVICE float GradientMagnitude(Volume& V, const Vec3f& P)
{
	const Vec3f HalfSpacing = 0.5f / V.GetSpacing();

	float D = 0.0f, Sum = 0.0f;

	D = (V.GetIntensity(P + Vec3f(V.GetSpacing()[0], 0.0f, 0.0f)) - V.GetIntensity(P - Vec3f(V.GetSpacing()[0], 0.0f, 0.0f))) * 0.5f;
	Sum += D * D;

	D = (V.GetIntensity(P + Vec3f(0.0f, V.GetSpacing()[1], 0.0f)) - V.GetIntensity(P - Vec3f(0.0f, V.GetSpacing()[1], 0.0f))) * 0.5f;
	Sum += D * D;

	D = (V.GetIntensity(P + Vec3f(0.0f, 0.0f, V.GetSpacing()[2])) - V.GetIntensity(P - Vec3f(0.0f, 0.0f, V.GetSpacing()[2]))) * 0.5f;
	Sum += D * D;

	return sqrtf(Sum);
}

}
