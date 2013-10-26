#pragma once

#include "core\kernel.cuh"

class Renderer;

namespace ExposureRender
{

extern "C" void Filter(dim3 Grid, dim3 Block, Renderer* HostRenderer, Renderer* DevRenderer);

}