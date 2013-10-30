
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

	qDebug() << SocketDescriptor << "client connected";
}

QGuiSocket::~QGuiSocket()
{
}

void QGuiSocket::OnReceiveData(const QString& Action, QDataStream& DataStream)
{
	if (Action == "VOLUME" || Action == "BITMAP")
	{
		this->SaveResource(DataStream);

		

		// this->RendererServer->SendDataToAll("VOLUME", ByteArrayOut);
	}
}