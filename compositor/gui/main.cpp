
#include "server\rendererserver.h"
#include "server\guiserver.h"
#include "gui\compositorwindow.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include <QtGui>

int main(int argc, char **argv)
{
	qDebug() << "Starting up compositor";

	QSettings Settings("compositor.ini", QSettings::IniFormat);

	QApplication Application(argc, argv);
	
	Application.setApplicationName("Exposure Render Compositor");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	QRendererServer RendererServer;
	QGuiServer GuiServer;

	QCompositorWindow CompositorWindow(&RendererServer, &GuiServer);

	if (Settings.value("gui/enabled", true).toBool())
	{
		CompositorWindow.show();
		CompositorWindow.resize(640, 480);
	}

	RendererServer.Start();
	GuiServer.Start();

    return Application.exec();
}
