
#include "server\guiserver.h"
#include "socket\guisocket.h"

#include <QDebug>

#include <QtGui>

QGuiServer::QGuiServer(QObject* Parent /*= 0*/) :
	QBaseServer("Gui", Parent),
	Settings("compositor.ini", QSettings::IniFormat)
{
	this->ListenPort = Settings.value("network/guiport", 6000).toInt();
}

void QGuiServer::OnNewConnection(const int& SocketDescriptor)
{
	QGuiSocket* GuiSocket = new QGuiSocket(SocketDescriptor, this);
	
	this->Connections.append(GuiSocket);
}