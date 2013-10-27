#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void Estimate(Renderer* HostRenderer, Renderer* DevRenderer);

}