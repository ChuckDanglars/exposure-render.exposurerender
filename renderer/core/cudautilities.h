/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

class CudaTimer
{
public:
	CudaTimer(void);
	virtual ~CudaTimer(void);

	void	StartTimer(void);
	float	StopTimer(void);
	float	ElapsedTime(void);

private:
	bool			m_Started;
	cudaEvent_t 	m_EventStart;
	cudaEvent_t 	m_EventStop;
};

bool InitializeCuda(void);
void HandleCudaError(const cudaError_t CudaError, const char* pDescription = "");
void HandleCudaKernelError(const cudaError_t CudaError, const char* pName = "");
int GetTotalCudaMemory(void);
int GetAvailableCudaMemory(void);
int GetUsedCudaMemory(void);
int GetMaxGigaFlopsDeviceID(void);
bool SetCudaDevice(const int& CudaDeviceID);
void ResetDevice(void);