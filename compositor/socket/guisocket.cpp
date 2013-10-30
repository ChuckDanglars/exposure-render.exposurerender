
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
	if (Action == "VOLUME")
	{
		QByteArray ByteArray;
		QString FileName;

		DataStream >> FileName;
		DataStream >> ByteArray;

		QFile File("resources//" + FileName);
		File.open(QIODevice::WriteOnly);
		File.writeBlock(ByteArray);
		File.close();

		qDebug() << "Received" << Action << FileName << "of" << ByteArray.count() << "bytes";

		QByteArray ByteArrayOut;
		QDataStream DataStreamOut(&ByteArrayOut, QIODevice::WriteOnly);

		DataStreamOut << FileName;
		DataStreamOut << ByteArray;

		this->RendererServer->SendDataToAll("VOLUME", ByteArrayOut);
	}
}