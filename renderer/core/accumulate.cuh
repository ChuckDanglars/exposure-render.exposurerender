#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void Accumulate(Renderer* HostRenderer, Renderer* DevRenderer);

}