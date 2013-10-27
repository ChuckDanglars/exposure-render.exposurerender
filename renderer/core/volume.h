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
#include "texture\cudatexture3d.h"
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
		Spacing(1.0f),
		InvSpacing(1.0f),
		Size(1.0f),
		InvSize(1.0f),
		MinStep(1.0f),
		Voxels(),
		AcceleratorType(Enums::Octree),
		MaxGradientMagnitude(0.0f),
		Tracer()
	{
	}

	/*! Computes the gradient at \a P using central differences
		@param[in] P Position at which to compute the gradient
		@return Gradient at \a P
	*/
	HOST void Set(const Vec3i& Resolution, const Vec3f& Spacing, short* Data, const Matrix44& Matrix = Matrix44())
	{
		this->Transform.Set(Matrix);

		this->Voxels.Resize(Resolution);
		this->Voxels.FromHost(Data);

		this->Spacing		= Spacing;
		this->InvSpacing	= 1.0f / this->Spacing;
		this->Size			= Vec3f(Resolution[0] * this->Spacing[0], Resolution[1] * this->Spacing[1], Resolution[2] * this->Spacing[2]);
		this->InvSize		= 1.0f / this->Size;

		this->BoundingBox.SetMinP(Vec3f(0.0f));
		this->BoundingBox.SetMaxP(this->Size);
	}
	
	/*! Gets the voxel data at \a P
		@param[in] P Position
		@return Data at \a XYZ volume
	*/
	DEVICE unsigned short GetIntensity(const Vec3f& P)
	{
		return this->Voxels(Vec3i(P[0] * this->Spacing[0], P[1] * this->Spacing[1], P[2] * this->Spacing[2]));
	}

	/*! Computes the gradient at \a P using central differences
		@param[in] P Position at which to compute the gradient
		@return Gradient at \a P
	*/
	DEVICE Vec3f GradientCD(const Vec3f& P)
	{
		const float Intensity[3][2] = 
		{
			{ this->GetIntensity(P + Vec3f(this->Spacing[0], 0.0f, 0.0f)), this->GetIntensity(P - Vec3f(this->Spacing[0], 0.0f, 0.0f)) },
			{ this->GetIntensity(P + Vec3f(0.0f, this->Spacing[1], 0.0f)), this->GetIntensity(P - Vec3f(0.0f, this->Spacing[1], 0.0f)) },
			{ this->GetIntensity(P + Vec3f(0.0f, 0.0f, this->Spacing[2])), this->GetIntensity(P - Vec3f(0.0f, 0.0f, this->Spacing[2])) }
		};

		return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
	}
	
	/*! Computes the gradient at \a P using forward differences
		@param[in] P Position at which to compute the gradient
		@return Gradient at \a P
	*/
	DEVICE Vec3f GradientFD(const Vec3f& P)
	{
		const float Intensity[4] = 
		{
			this->GetIntensity(P),
			this->GetIntensity(P + Vec3f(this->Spacing[0], 0.0f, 0.0f)),
			this->GetIntensity(P + Vec3f(0.0f, this->Spacing[1], 0.0f)),
			this->GetIntensity(P + Vec3f(0.0f, 0.0f, this->Spacing[2]))
		};

		return Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
	}
	
	/*! Computes the filtered gradient at \a P using central differences
		@param[in] P Position at which to compute the gradient
		@return Gradient at \a P
	*/
	DEVICE Vec3f GradientFiltered(const Vec3f& P)
	{
		Vec3f Offset(Vec3f(this->Spacing[0], 0.0f, 0.0f)[0], Vec3f(0.0f, this->Spacing[1], 0.0f)[1], Vec3f(0.0f, 0.0f, this->Spacing[2])[2]);

		const Vec3f G0 = GradientCD(P);
		const Vec3f G1 = GradientCD(P + Vec3f(-Offset[0], -Offset[1], -Offset[2]));
		const Vec3f G2 = GradientCD(P + Vec3f( Offset[0],  Offset[1],  Offset[2]));
		const Vec3f G3 = GradientCD(P + Vec3f(-Offset[0],  Offset[1], -Offset[2]));
		const Vec3f G4 = GradientCD(P + Vec3f( Offset[0], -Offset[1],  Offset[2]));
		const Vec3f G5 = GradientCD(P + Vec3f(-Offset[0], -Offset[1],  Offset[2]));
		const Vec3f G6 = GradientCD(P + Vec3f( Offset[0],  Offset[1], -Offset[2]));
		const Vec3f G7 = GradientCD(P + Vec3f(-Offset[0],  Offset[1],  Offset[2]));
		const Vec3f G8 = GradientCD(P + Vec3f( Offset[0], -Offset[1], -Offset[2]));
	    
		const Vec3f L0 = Lerp(0.5f, Lerp(0.5f, G1, G2), Lerp(0.5f, G3, G4));
		const Vec3f L1 = Lerp(0.5f, Lerp(0.5f, G5, G6), Lerp(0.5f, G7, G8));
	    
		return Lerp(0.75f, G0, Lerp(0.5f, L0, L1));
	}
	
	/*! Computes the gradient at \a P using \a GradientMode
		@param[in] P Position at which to compute the gradient
		@param[in] GradientMode Type of gradient computation
		@return Gradient at \a P
	*/
	DEVICE Vec3f Gradient(const Vec3f& P, const Enums::GradientMode& GradientMode)
	{
		switch (GradientMode)
		{
			case Enums::ForwardDifferences:		return GradientFD(P);
			case Enums::CentralDifferences:		return GradientCD(P);
			case Enums::Filtered:				return GradientFiltered(P);
		}

		return GradientFD(P);
	}
	
	/*! Computes the normalized gradient at \a P using \a GradientMode
		@param[in] P Position at which to compute the gradient
		@param[in] GradientMode Type of gradient computation
		@return Gradient at \a P
	*/
	DEVICE Vec3f NormalizedGradient(const Vec3f& P, const Enums::GradientMode& GradientMode)
	{
		return Normalize(Gradient(P, GradientMode));
	}
	
	/*! Computes the gradient magnitude at \a P
		@param[in] P Position at which to compute the gradient magnitude
		@return Gradient magnitude at \a P
	*/
	DEVICE float GradientMagnitude(const Vec3f& P)
	{
		const Vec3f HalfSpacing = 0.5f / this->Spacing;

		float D = 0.0f, Sum = 0.0f;

		D = (this->GetIntensity(P + Vec3f(this->Spacing[0], 0.0f, 0.0f)) - this->GetIntensity(P - Vec3f(this->Spacing[0], 0.0f, 0.0f))) * 0.5f;
		Sum += D * D;

		D = (this->GetIntensity(P + Vec3f(0.0f, this->Spacing[1], 0.0f)) - this->GetIntensity(P - Vec3f(0.0f, this->Spacing[1], 0.0f))) * 0.5f;
		Sum += D * D;

		D = (this->GetIntensity(P + Vec3f(0.0f, 0.0f, this->Spacing[2])) - this->GetIntensity(P - Vec3f(0.0f, 0.0f, this->Spacing[2]))) * 0.5f;
		Sum += D * D;

		return sqrtf(Sum);
	}
	
	/*! Intersects the volume with ray \a R and determine if a scattering event \a SE occurs within the volume
		@param[in] R Ray to intersect the volume with
		@param[in] RNG Random number generator
		@param[in] SE Scattering event, filled if a scattering event occurs
		@return Whether a scattering event has occured or not
	*/
	DEVICE bool Intersect(Ray R, RNG& RNG, ScatterEvent& SE)
	{
		if (!this->BoundingBox.Intersect(R, R.MinT, R.MaxT))
			return false;

		const float S	= -log(RNG.Get1()) / this->Tracer.GetDensityScale();
		float Sum		= 0.0f;

		R.MinT += RNG.Get1() * this->Tracer.GetStepFactorPrimary();

		while (Sum < S)
		{
			if (R.MinT + this->Tracer.GetStepFactorPrimary() >= R.MaxT)
				return false;
		
			SE.SetP(R(R.MinT));
			SE.SetIntensity(this->Voxels(SE.GetP()));

			Sum		+= this->Tracer.GetDensityScale() * this->Tracer.GetOpacity(SE.GetIntensity()) * this->Tracer.GetStepFactorPrimary();
			R.MinT	+= this->Tracer.GetStepFactorPrimary();
		}

		SE.SetWo(-R.D);
		SE.SetN(this->NormalizedGradient(SE.GetP(), Enums::CentralDifferences));
		SE.SetT(R.MinT);
		SE.SetScatterType(Enums::Volume);

		return true;
	}

	/*! Determine if a scattering event occurs within the parametric range of the ray \a R
		@param[in] R Ray to intersect the volume with
		@param[in] RNG Random number generator
		@return Whether a scattering event has occured or not
	*/
	DEVICE bool Occlusion(Ray R, RNG& RNG)
	{
		if (!this->Tracer.GetShadows())
			return false;

		float MaxT = 0.0f;

		if (!this->BoundingBox.Intersect(R, R.MinT, MaxT))
			return false;

		R.MaxT = min(R.MaxT, MaxT);

		const float S	= -log(RNG.Get1()) / this->Tracer.GetDensityScale();
		float Sum		= 0.0f;
	
		R.MinT += RNG.Get1() * this->Tracer.GetStepFactorOcclusion();

		while (Sum < S)
		{
			if (R.MinT > R.MaxT)
				return false;

			Sum		+= this->Tracer.GetDensityScale() * this->Tracer.GetOpacity(this->Voxels(R(R.MinT))) * this->Tracer.GetStepFactorOcclusion();
			R.MinT	+= this->Tracer.GetStepFactorOcclusion();
		}

		return true;
	}

	Transform						Transform;					/*! Transform of the volume */
	BoundingBox						BoundingBox;				/*! Encompassing bounding box */
	Vec3f							Spacing;					/*! Voxel spacing */
	Vec3f							InvSpacing;					/*! Inverse voxel spacing */
	Vec3f							Size;						/*! Volume size */
	Vec3f							InvSize;					/*! Inverse volume size */
	float							MinStep;					/*! Minimum step size */
	CudaBuffer3D<short>				Voxels;						/*! Voxel 3D buffer */
	Enums::AcceleratorType			AcceleratorType;			/*! Type of ray traversal accelerator */
	float							MaxGradientMagnitude;		/*! Maximum gradient magnitude */
	Tracer							Tracer;						/*! Tracer */
};

}
