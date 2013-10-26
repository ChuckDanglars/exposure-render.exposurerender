#pragma once

#include "buffer\buffers.h"

namespace ExposureRender
{

extern "C" void Render(float Position[3], float FocalPoint[3], float ViewUp[3], const int& Width, const int& Height, unsigned char* Output);

}