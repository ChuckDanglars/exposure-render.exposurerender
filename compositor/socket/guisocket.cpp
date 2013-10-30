
#include "guisocket.h"
#include "server\rendererserver.h"

#include <QFile>

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
	qDebug() << __FUNCTION__;

	if (Action == "VOLUME" || Action == "BITMAP")
	{
		QByteArray ByteArray;
		QString FileName;

		DataStream >> FileName;
		DataStream >> ByteArray;

		this->RendererServer->SendDataToAll(Action, ByteArray);

		QFile File(FileName);
		File.open(QIODevice::WriteOnly);
		File.writeBlock(ByteArray);
	}
}