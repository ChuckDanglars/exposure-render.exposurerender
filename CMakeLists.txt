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
	gpujpeg
	gui
	renderer
	utilities)

ADD_SUBDIRECTORY(compositor)
ADD_SUBDIRECTORY(gpujpeg)
ADD_SUBDIRECTORY(gui)
ADD_SUBDIRECTORY(renderer)
ADD_SUBDIRECTORY(utilities)

### INSTALLER ###
MESSAGE(STATUS "Creating installer project")

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

# Shortcut in start menu
SET(CPACK_PACKAGE_EXECUTABLES "ExposureRender;Exposure Render")

INCLUDE(CPack)


