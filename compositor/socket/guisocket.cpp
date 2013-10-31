
#include "guisocket.h"
#include "server\rendererserver.h"

#include <QFile>
#include <QDebug>

QGuiSocket::QGuiSocket(int SocketDescriptor, QRendererServer* RendererServer, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	RendererServer(RendererServer)
{
	if (!this->setSocketDescriptor(SocketDescriptor))
		return;

	qDebug() << SocketDescriptor << "gui connected";
}

QGuiSocket::~QGuiSocket()
{
}

void QGuiSocket::OnReceiveData(const QString& Action, QByteArray& ByteArray)
{
	if (Action == "VOLUME" || Action == "BITMAP")
	{
		qDebug() << "Received" << Action.toLower();

		this->SaveResource(ByteArray);
		this->RendererServer->SendDataToAll(Action, ByteArray);
	}
}