#pragma once

#include "core\kernel.cuh"

namespace ExposureRender
{

class Renderer;

extern "C" void ToneMap(Renderer* HostRenderer, Renderer* DevRenderer);

}