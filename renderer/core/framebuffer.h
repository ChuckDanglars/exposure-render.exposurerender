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
#include "color\color.h"

namespace ExposureRender
{

/*! Frame buffer class */
class FrameBuffer
{
public:
	HOST FrameBuffer(void) :
		Resolution(),
		IterationEstimate(),
		AccumulatedEstimate(),
		RunningEstimate(),
		RandomSeeds1(),
		RandomSeeds2(),
		StartRandomSeeds1(),
		StartRandomSeeds2()
	{
		this->Resize(Vec2i(1024, 768));
	}

	HOST void Resize(const Vec2i& Resolution)
	{
		
		if (this->Resolution == Resolution)
			return;
		
		this->Resolution = Resolution;

		this->IterationEstimate.Resize(this->Resolution);
		this->AccumulatedEstimate.Resize(this->Resolution);
		this->RunningEstimate.Resize(this->Resolution);
		this->RandomSeeds1.Resize(this->Resolution);
		this->RandomSeeds2.Resize(this->Resolution);
		this->StartRandomSeeds1.Resize(this->Resolution);
		this->StartRandomSeeds2.Resize(this->Resolution);
		
		this->RandomSeeds1.FromHost(this->StartRandomSeeds1.GetData());
		this->RandomSeeds2.FromHost(this->StartRandomSeeds2.GetData());
	}

	Vec2i							Resolution;					/*! Resolution of the frame buffer */
	CudaBuffer2D<ColorXYZAf>		IterationEstimate;			/*! Estimate from a single iteration of the mc algorithm */
	CudaBuffer2D<ColorRGBAul>		AccumulatedEstimate;		/*! Accumulation buffer */
	CudaBuffer2D<ColorRGBAuc>		RunningEstimate;			/*! Integrated estimate */
	CudaRandomSeedBuffer2D			RandomSeeds1;				/*! Random seed buffer */
	CudaRandomSeedBuffer2D			RandomSeeds2;				/*! Random seed buffer */
	HostRandomSeedBuffer2D			StartRandomSeeds1;			/*! Random seed buffer */
	HostRandomSeedBuffer2D			StartRandomSeeds2;			/*! Random seed buffer */
};

}
