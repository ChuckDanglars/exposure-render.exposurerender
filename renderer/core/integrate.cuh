#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void Integrate(Renderer* HostRenderer, Renderer* DevRenderer);

}