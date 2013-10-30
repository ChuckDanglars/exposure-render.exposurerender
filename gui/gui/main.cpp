
#include "guiwindow.h"
#include "utilities\gui\renderoutputwidget.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include <QtGui>

int main(int argc, char **argv)
{
	qDebug() << "Starting up Exposure Render Graphical User Interface";

	QApplication Application(argc, argv);
	
	Application.setApplicationName("Exposure Render Compositor");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	
	/*
	QGuiWindow CompositorWindow(&Server);
	
    CompositorWindow.show();
	CompositorWindow.resize(640, 480);
	
	Server.Start();

	QObject::connect(CompositorWindow.GetRenderOutputWidget(), SIGNAL(CameraUpdate(float*,float*,float*)), &Server, SLOT(OnCameraUpdate(float*,float*,float*)));
	*/

    return Application.exec();
}
