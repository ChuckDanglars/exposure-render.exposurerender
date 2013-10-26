
// #include "Stable.h"

#include "cudautilities.h"

#include <algorithm>

#include <QString>
#include <QDebug>

CudaTimer::CudaTimer(void)
{
	StartTimer();
}

CudaTimer::~CudaTimer(void)
{
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);
}

void CudaTimer::StartTimer(void)
{
	cudaEventCreate(&m_EventStart);
	cudaEventCreate(&m_EventStop);
	cudaEventRecord(m_EventStart, 0);

	m_Started = true;
}

float CudaTimer::StopTimer(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);

	m_Started = false;

	return TimeDelta;
}

float CudaTimer::ElapsedTime(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);

	m_Started = false;

	return TimeDelta;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void GetCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error = 	cuDeviceGetAttribute( attribute, device_attribute, device );

	if( CUDA_SUCCESS != error) {
		fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);
		exit(-1);
	}
}

void HandleCudaError(const cudaError_t CudaError, const char* pDescription /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	qDebug() << "Encountered a critical CUDA error: " << QString(pDescription) << " " << QString(cudaGetErrorString(CudaError));

	throw new QString("Encountered a critical CUDA error: " + QString(pDescription) + " " + QString(cudaGetErrorString(CudaError)));
}

void HandleCudaKernelError(const cudaError_t CudaError, const char* pName /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	qDebug() << "The '" << QString::fromAscii(pName) << "' kernel caused the following CUDA runtime error: " << QString(cudaGetErrorString(CudaError));

	throw new QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError)));
}

int GetTotalCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total;
}

int GetAvailableCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Available;
}

int GetUsedCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total - Available;
}

int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 },
	  { 0x11,  8 },
	  { 0x12,  8 },
	  { 0x13,  8 },
	  { 0x20, 32 },
	  { 0x21, 48 },
	  {   -1, -1 } 
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}

int GetMaxGigaFlopsDeviceID(void)
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = std::max(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

bool SetCudaDevice(const int& CudaDeviceID)
{
	const cudaError_t CudaError = cudaSetDevice(CudaDeviceID);

	HandleCudaError(CudaError, "set Cuda device");

	return CudaError == cudaSuccess;
}

void ResetDevice(void)
{
	HandleCudaError(cudaDeviceReset(), "reset device");
}