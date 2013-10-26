#pragma once

#include "buffer\buffers.h"
#include "color\color.h"

namespace ExposureRender
{

extern "C" float Combine(int Width, int Height, unsigned char* Estimates[20], const int& NoEstimates, unsigned char* Estimate);

}