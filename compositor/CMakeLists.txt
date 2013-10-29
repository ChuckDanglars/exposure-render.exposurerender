MESSAGE(STATUS "Creating compositor project")

MESSAGE(STATUS "Finding qt package")
FIND_PACKAGE(QT4 REQUIRED)
SET(QT_USE_QTNETWORK TRUE)
SET(QT_USE_QTOPENGL TRUE)
FIND_PACKAGE(OPENGL REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
ADD_DEFINITIONS(-DQT_GUI_LIBS -DQT_CORE_LIB -DQT_XML_LIB -DQT3_SUPPORT)

MESSAGE(STATUS "Finding cuda package")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
SET(CUDA_VERBOSE_BUILD ON)

FILE(GLOB CompositorSources "*.h" "*.cpp")
FILE(GLOB CompositorHeaders "*.h")
FILE(GLOB CompositorCudaSources "*.cuh" "*.cu")

QT4_WRAP_CPP(CompositorHeadersMoc ${CompositorHeaders})
CUDA_ADD_EXECUTABLE(Compositor ${CompositorSources} ${CompositorCudaSources} ${CompositorHeadersMoc})
TARGET_LINK_LIBRARIES(Compositor Utilities Opengl32)

INSTALL_TARGETS(/bin Compositor)