#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void Acceleration(Renderer* HostRenderer, Renderer* DevRenderer);

}