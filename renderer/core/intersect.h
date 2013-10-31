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

/*! Intersects volume \a V with ray \a R and determine if a scattering event \a SE occurs within the volume
	@param[in] V Input volume
	@param[in] R Ray in world space to intersect the volume with
	@param[in] RNG Random number generator
	@param[in] SE Scattering event, filled if a scattering event occurs
	@return Whether a scattering event has occured or not
*/
DEVICE bool IntersectVolume(Volume& V, Ray R, RNG& RNG, ScatterEvent& SE)
{
	if (!V.GetBoundingBox().Intersect(R, R.MinT, R.MaxT))
		return false;
	else
		return true;

	Tracer& T = V.GetTracer();

	const float S	= -log(RNG.Get1()) / T.GetDensityScale();
	float Sum		= 0.0f;
		
	R.MinT += RNG.Get1() * T.GetStepFactorPrimary();

	Vec3f P;
	short Intensity = 0;

	while (Sum < S)
	{
		if (R.MinT + T.GetStepFactorPrimary() >= R.MaxT)
			return false;
		
		P			= R(R.MinT);
		Intensity	= V.GetIntensity(P);

		Sum		+= T.GetDensityScale() * T.GetStepFactorPrimary();
		R.MinT	+= T.GetStepFactorPrimary();
	}

	SE.SetP(P);
	SE.SetIntensity(Intensity);
	SE.SetWo(-R.D);
	// SE.SetN(V.NormalizedGradient(SE.GetP(), Enums::CentralDifferences));
	SE.SetT(R.MinT);
	SE.SetScatterType(Enums::Volume);

	return true;
}

/*! Determine if a scattering event occurs with volume \a V within the parametric range of the ray \a R
	@param[in] V Input volume
	@param[in] R Ray in world space to intersect the volume with
	@param[in] RNG Random number generator
	@return Whether a scattering event has occured or not
*/
DEVICE bool IntersectP(Volume& V, Ray R, RNG& RNG)
{
	Tracer& T = V.GetTracer();

	if (!T.GetShadows())
		return false;

	float MaxT = 0.0f;

	if (!V.GetBoundingBox().Intersect(R, R.MinT, MaxT))
		return false;

	R.MaxT = min(R.MaxT, MaxT);

	const float S	= -log(RNG.Get1()) / T.GetDensityScale();
	float Sum		= 0.0f;
	
	R.MinT += RNG.Get1() * T.GetStepFactorOcclusion();

	while (Sum < S)
	{
		if (R.MinT > R.MaxT)
			return false;

		Sum		+= T.GetDensityScale() * T.GetOpacity(V.GetIntensity(R(R.MinT))) * T.GetStepFactorOcclusion();
		R.MinT	+= T.GetStepFactorOcclusion();
	}

	return true;
}

}
