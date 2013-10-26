#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void Filter(Renderer* HostRenderer, Renderer* DevRenderer);

}