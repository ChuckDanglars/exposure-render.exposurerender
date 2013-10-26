
#include "server.h"
#include "compositorwindow.h"
#include "utilities\renderoutputwidget.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include <QtGui>

#include "renderer\vector\vector.h"
#include "renderer\color\color.h"

int main(int argc, char **argv)
{
	qDebug() << "Starting up compositor";

	QApplication Application(argc, argv);
	
	Application.setApplicationName("Exposure Render Compositor");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	QServer Server;

	QCompositorWindow CompositorWindow(&Server);
	
    CompositorWindow.show();
	CompositorWindow.resize(640, 480);
	
	Server.Start();

	QObject::connect(CompositorWindow.GetRenderOutputWidget(), SIGNAL(CameraUpdate(float*,float*,float*)), &Server, SLOT(OnCameraUpdate(float*,float*,float*)));

    return Application.exec();
}
