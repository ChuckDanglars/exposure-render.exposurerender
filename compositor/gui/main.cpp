
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
	qDebug() << "Starting up Exposure Render compositor";

	QSettings Settings("compositor.ini", QSettings::IniFormat);

	QApplication Application(argc, argv);
	
	Application.setApplicationName("Exposure Render Compositor");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	QRendererServer RendererServer;
	QGuiServer GuiServer(&RendererServer);

	QCompositorWindow CompositorWindow(&RendererServer, &GuiServer);

	const bool& GuiEnabled = Settings.value("gui/enabled", true).toBool();

	qDebug() << "Gui is" << (GuiEnabled ? "enabled" : "disabled");

	if (GuiEnabled)
	{
		CompositorWindow.show();
		CompositorWindow.resize(640, 480);
	}

	RendererServer.Start();
	GuiServer.Start();

    return Application.exec();
}
