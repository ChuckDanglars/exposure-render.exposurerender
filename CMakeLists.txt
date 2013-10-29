CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7 FATAL_ERROR)

PROJECT(ExposureRender)

MESSAGE(STATUS "Creating exposure render build environment")

SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_EXPORTING")

INCLUDE_DIRECTORIES(
	${QT_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
	compositor
	utilities
	gpujpeg
	renderer)

ADD_SUBDIRECTORY(compositor)
ADD_SUBDIRECTORY(gpujpeg)
ADD_SUBDIRECTORY(gui)
ADD_SUBDIRECTORY(renderer)
ADD_SUBDIRECTORY(utilities)

### INSTALLER ###
MESSAGE(STATUS "Creating installer project")

### VTK dependencies ###
#FILE(GLOB VtkDlls "${VTK_DIR}/bin/RelWithDebInfo/*.dll")
#INSTALL_FILES(/bin FILES ${VtkDlls})

### CUDA dependencies ###
SET(CudaDlls ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32_55.dll)
INSTALL_FILES(/bin FILES ${CudaDlls})

### QT dependencies ###
SET(QtDlls ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtXml4.dll ${QT_BINARY_DIR}/QtSql4.dll ${QT_BINARY_DIR}/QtNetwork4.dll ${QT_BINARY_DIR}/QtOpenGl4.dll)
INSTALL_FILES(/bin FILES ${QtDlls})

### Configuration files ###
FILE(GLOB Presets "${CMAKE_CURRENT_SOURCE_DIR}/*.ini")
INSTALL_FILES(/bin FILES ${Presets})

# Requires re-distributables
INCLUDE(InstallRequiredSystemLibraries)

### Configure NSIS ###
SET(CPACK_GENERATOR NSIS)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Exposure Render")
SET(CPACK_PACKAGE_VENDOR "TU Delft")
SET(CPACK_PACKAGE_DEFAULT_LOCATION /)
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "0")

# Menu links
SET(CPACK_NSIS_MENU_LINKS "http://graphics.tudelft.nl/" "TU Delft Graphics Website" "http://code.google.com/p/exposure-render/" "Exposure Render Website" "http://code.google.com/p/exposure-render/downloads" "Data sets" "http://code.google.com/p/exposure-render/wiki" "Wiki")

# Icon in the add/remove control panel, this must be an .exe file 
SET(CPACK_NSIS_INSTALLED_ICON_NAME ${EXE_TARGET_NAME}.exe)

# Install directory
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "Exposure Render")

# SET(SIDEBAR_IMAGE ${CMAKE_SOURCE_DIR}/bonsai.bmp}
# SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "!define MUI_WELCOMEFINISHPAGE_BITMAP \\\"${SIDEBAR_IMAGE}\\\")

# SET(CPACK_PACKAGE_ICON "c:\\WindowsMovieMaker.bmp")
SET(CPACK_NSIS_PACKAGE_NAME "Exposure Render")
SET(CPACK_NSIS_DISPLAY_NAME "Exposure Render")
SET(CPACK_NSIS_CONTACT "t.kroes@tudelft.nl")
SET(CPACK_NSIS_HELP_LINK "http://graphics.tudelft.nl/")
SET(CPACK_NSIS_URL_INFO_ABOUT "http://code.google.com/p/exposure-render/")
SET(CPACK_NSIS_MUI_FINISHPAGE_RUN "ExposureRender")

SET(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin")

# Shortcut in start menu
SET(CPACK_PACKAGE_EXECUTABLES "ExposureRender;Exposure Render")

INCLUDE(CPack)


