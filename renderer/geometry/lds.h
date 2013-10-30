
#pragma once

#include "utilities\general\define.h"

namespace ExposureRender
{

class Halton1D
{
public:
	HOST_DEVICE Halton1D(int Base = 2, unsigned int Offset = 0)
	{
		this->Base		= Base;
        this->InvBase	= 1.0f / (float)this->Base;
		this->Value		= 0.0f;

        this->SetOffset(Offset);
	}

    HOST_DEVICE void SetOffset(unsigned int Offset)
    {
		float Factor = this->InvBase;

		this->Value = 0.0f;
		
        while (Offset > 0)
		{
			float div = Offset / this->Base;
			float num = Offset - (floor(div) * this->Base);
            this->Value += (float)(num) * Factor;
            Offset /= this->Base;
            Factor *= this->InvBase;
		}
	}

    HOST_DEVICE void Reset()
	{
		this->Value = 0.0f;
	}

    HOST_DEVICE float GetNext()
	{
        float r = 0.999999f - this->Value;

        if (this->InvBase < r)
		{
            this->Value += this->InvBase;
		}
        else
		{
            float hh, h	= this->InvBase;

            while (h >= r)
			{
                hh = h;
                h *= this->InvBase;
			}
			

            this->Value += hh + h - 1.0f;
		}

		return this->Value;
	}

private:
	unsigned int	Base;
	float			InvBase;
	float			Value;
};

class Halton2D
{
public:
	HOST_DEVICE Halton2D(int Offset = 0) :
		Halton1(2, Offset),
		Halton2(3, Offset)
	{
	}

	HOST_DEVICE float* GetNext()
	{
		this->Value[0] = this->Halton1.GetNext();
		this->Value[1] = this->Halton2.GetNext();

		return this->Value;
	}

private:
	Halton1D	Halton1;
	Halton1D	Halton2;
	float		Value[2];
};

class Halton3D
{
public:
	HOST_DEVICE Halton3D(int Offset = 0) :
		Halton1(2, Offset),
		Halton2(3, Offset),
		Halton3(5, Offset)
	{
	}

	HOST_DEVICE float* GetNext()
	{
		this->Value[0] = this->Halton1.GetNext();
		this->Value[1] = this->Halton2.GetNext();
		this->Value[2] = this->Halton3.GetNext();

		return this->Value;
	}

private:
	Halton1D	Halton1;
	Halton1D	Halton2;
	Halton1D	Halton3;
	float		Value[3];
};

}