MESSAGE(STATUS "Creating gpu jpeg project")

MESSAGE(STATUS "Finding cuda package")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
SET(CUDA_VERBOSE_BUILD ON)

FILE(GLOB GpuJpegSources "*.h" "*.cpp" "*.cuh" "*.cu")
CUDA_ADD_LIBRARY(GpuJpeg ${GpuJpegSources})

#INSTALL_TARGETS(/bin GpuJpeg)