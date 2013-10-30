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

FILE(GLOB CudaSources "cuda/*.cuh" "cuda/*.cu")
FILE(GLOB GuiSources "gui/*.h" "gui/*.cpp")
FILE(GLOB ServerSources "server/*.h" "server/*.cpp")
FILE(GLOB SocketSources "socket/*.h" "socket/*.cpp")

SOURCE_GROUP("cuda" FILES ${CudaSources})
SOURCE_GROUP("gui" FILES ${GuiSources})
SOURCE_GROUP("server" FILES ${ServerSources})
SOURCE_GROUP("socket" FILES ${SocketSources})

QT4_WRAP_CPP(CompositorHeadersMoc gui/compositorwindow.h server/rendererserver.h server/guiserver.h socket/guisocket.h socket/renderersocket.h)
CUDA_ADD_EXECUTABLE(Compositor ${CudaSources} ${GuiSources} ${ServerSources} ${SocketSources} ${CompositorHeadersMoc})
TARGET_LINK_LIBRARIES(Compositor Utilities Opengl32)

INSTALL_TARGETS(/bin Compositor)