
#include "render.cuh"
#include "core\estimate.cuh"
#include "core\tonemap.cuh"
#include "core\filter.cuh"
#include "core\integrate.cuh"
#include "core\accumulate.cuh"
#include "core\camera.h"

namespace ExposureRender
{

void Render(Camera& HostCamera)
{
	
	Estimate(HostCamera);/*
	ToneMap(HostCamera);
	Filter(HostCamera);
	Accumulate(HostCamera);
	Integrate(HostCamera);
	*/
}

}
