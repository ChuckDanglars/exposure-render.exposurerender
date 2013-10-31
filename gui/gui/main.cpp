
#include "guiwindow.h"
#include "network\compositorsocket.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include <QtGui>

int main(int argc, char **argv)
{
	qDebug() << "Starting up Exposure Render Graphical User Interface";

	QSettings Settings("gui.ini", QSettings::IniFormat);

	QApplication Application(argc, argv);
	
	Application.setApplicationName("Exposure Render - GUI");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	if (!QDir("resources").exists())
	{
		qDebug() << "Resource directory does not exist, creating it";
		
		QDir().mkdir("resources");
	}

	QCompositorSocket CompositorSocket(&Application);
	
	const int Wait		= Settings.value("network/wait", 2000).toInt();
	QString HostName	= Settings.value("network/host", "localhost").toString();
	const quint16 Port	= Settings.value("network/port", 6001).toInt();

	qDebug() << "Connecting to" << HostName << "through port" << Port;

	CompositorSocket.connectToHost(HostName, Port);
	
	if (!CompositorSocket.waitForConnected(Wait))
	{
		qDebug() << "Unable to connect to host";

		qDebug() << "Last resort: trying to connect to compositor on localhost" << "through port" << Port;

		HostName = "localhost";

		CompositorSocket.connectToHost(HostName, Port);

		if (!CompositorSocket.waitForConnected(Wait))
			qDebug() << "Not connected to host";
	}

	if (CompositorSocket.isOpen())
		qDebug() << "Connected to compositor on" << HostName << "through port" << Port;

	QGuiWindow GuiWindow(&CompositorSocket);

	GuiWindow.show();
	GuiWindow.resize(640, 480);
	
    return Application.exec();
}
