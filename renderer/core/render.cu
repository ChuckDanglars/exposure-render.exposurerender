
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
	Film& Film = HostCamera.GetFilm();

	if (Film.GetNoEstimates() == 1)
	{
		Film.GetAccumulatedEstimate().Reset();

		Film.GetRandomSeeds1().FromHost(Film.GetHostRandomSeeds1().GetData());
		Film.GetRandomSeeds2().FromHost(Film.GetHostRandomSeeds2().GetData());
	}

	Estimate(HostCamera);
	ToneMap(HostCamera);
	Filter(HostCamera);
	Accumulate(HostCamera);
	Integrate(HostCamera);
}

}
