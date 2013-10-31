MESSAGE(STATUS "Creating gui project")

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

FILE(GLOB GuiSources "gui/*.h" "gui/*.cpp")
FILE(GLOB NetworkSources "network/*.h" "network/*.cpp")

SOURCE_GROUP("gui" FILES ${GuiSources})
SOURCE_GROUP("network" FILES ${NetworkSources})

QT4_WRAP_CPP(GuiHeadersMoc gui/guiwindow.h network/compositorsocket.h)
CUDA_ADD_EXECUTABLE(Gui ${GuiSources} ${NetworkSources} ${GuiHeadersMoc})
TARGET_LINK_LIBRARIES(Gui Utilities Opengl32)

INSTALL_TARGETS(/gui Gui)

SET(QtDlls ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtXml4.dll ${QT_BINARY_DIR}/QtSql4.dll ${QT_BINARY_DIR}/QtNetwork4.dll ${QT_BINARY_DIR}/QtOpenGl4.dll)
INSTALL_FILES(/gui FILES ${QtDlls})

INSTALL_FILES(/gui FILES ${CMAKE_CURRENT_SOURCE_DIR}/gui.ini)

INCLUDE(InstallRequiredSystemLibraries)