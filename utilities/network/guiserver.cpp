
#include "server\guiserver.h"
#include "socket\guisocket.h"

#include <QDebug>

#include <QtGui>

QGuiServer::QGuiServer(QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	Connections()
{
}

void QGuiServer::Start()
{
	const int Port = this->Settings.value("network/guiport", 6000).toInt();

	qDebug() << "Starting Exposure Render server";

	if (!this->listen(QHostAddress::Any, Port))
	{
		qDebug() << "Could not start server";
	}
	else
	{
		qDebug() << "Server listening to any ip on port" << Port;
	}
}

void QGuiServer::incomingConnection(int SocketDescriptor)
{
	qDebug() << SocketDescriptor << "connecting...";
	
	QGuiSocket* GuiSocket = new QGuiSocket(SocketDescriptor, this);
	
	this->Connections.append(GuiSocket);
}