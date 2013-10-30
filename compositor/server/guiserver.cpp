
#include "server\guiserver.h"
#include "server\rendererserver.h"
#include "socket\guisocket.h"

#include <QDebug>

QGuiServer::QGuiServer(QRendererServer* RendererServer, QObject* Parent /*= 0*/) :
	QBaseServer("Gui", Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	RendererServer(RendererServer)
{
	this->ListenPort = Settings.value("network/guiport", 6000).toInt();
}

void QGuiServer::OnNewConnection(const int& SocketDescriptor)
{
	QGuiSocket* GuiSocket = new QGuiSocket(SocketDescriptor, RendererServer, this);
	
	this->Connections.append(GuiSocket);
}

void QGuiServer::OnStarted()
{
}