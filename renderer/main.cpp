
#include "core\renderthread.h"
#include "network\clientsocket.h"
#include "gui\rendererwindow.h"
#include "geometry\lds.h"

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include <QtGui>
#include <QTcpSocket>

int main(int argc, char **argv)
{
	Halton1D H1(2);
	Halton1D H2(2, 100);

	for (int i = 0; i < 100; i++)
	{
		qDebug() << "H1 Sample" << i << ":" << H1.GetNext();
		qDebug() << "H2 Sample" << i << ":" << H2.GetNext();
	}

	qDebug() << "Starting up renderer";
	
	QApplication Application(argc, argv);

	Application.setApplicationName("Exposure Render Compositor");
	Application.setOrganizationName("Delft University of Technology, department of Computer Graphics and Visualization");

	QSettings Settings("renderer.ini", QSettings::IniFormat);

	QRenderer Renderer(&Application);

	QClientSocket ClientSocket(&Renderer, &Application);
	
	const int Wait		= Settings.value("network/wait", "localhost").toInt();
	QString HostName	= Settings.value("network/host", "localhost").toString();
	const quint16 Port	= Settings.value("network/port", 6000).toInt();

	qDebug() << "Connecting to" << HostName << "on port" << Port;

	ClientSocket.connectToHost(HostName, Port);
	
	if (!ClientSocket.waitForConnected(Wait))
	{
		qDebug() << "Unable to connect to host";

		qDebug() << "Last resort: trying to connect to localhost";

		HostName = "localhost";

		ClientSocket.connectToHost(HostName, Port);

		if (!ClientSocket.waitForConnected(Wait))
			qDebug() << "Not connected to host";
	}

	if (ClientSocket.isOpen())
		qDebug() << "Connected to" << HostName << "on port" << Port;

	QRendererWindow RendererWindow(&Renderer);
	
    RendererWindow.show();
	RendererWindow.resize(640, 480);

	return Application.exec();
}
