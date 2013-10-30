MESSAGE(STATUS "Creating utilities project")

MESSAGE(STATUS "Finding qt package")
FIND_PACKAGE(QT4 REQUIRED)
SET(QT_USE_QTNETWORK TRUE)
SET(QT_USE_QTOPENGL TRUE)
SET(QT_USE_QTXML TRUE)
FIND_PACKAGE(OPENGL REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
ADD_DEFINITIONS(-DQT_GUI_LIBS -DQT_CORE_LIB -DQT_XML_LIB -DQT3_SUPPORT)

FILE(GLOB GeneralSources "general/*.h" "general/*.cpp")
FILE(GLOB GpuJpegSources "gpujpeg/*.h" "gpujpeg/*.cpp")
FILE(GLOB GuiSources "gui/*.h" "gui/*.cpp")
FILE(GLOB NetworkSources "network/*.h" "network/*.cpp")

SOURCE_GROUP("general" FILES ${GeneralSources})
SOURCE_GROUP("gpujpeg" FILES ${GpuJpegSources})
SOURCE_GROUP("gui" FILES ${GuiSources})
SOURCE_GROUP("network" FILES ${NetworkSources})

QT4_WRAP_CPP(UtilitiesHeadersMoc general/hysteresis.h gpujpeg/gpujpeg.h gui/renderoutputwidget.h network/basesocket.h network/baseserver.h)
ADD_LIBRARY(Utilities ${GeneralSources} ${GpuJpegSources} ${GuiSources} ${NetworkSources} ${UtilitiesHeadersMoc})
TARGET_LINK_LIBRARIES(Utilities GpuJpeg ${QT_LIBRARIES})

INSTALL_TARGETS(/bin Utilities)