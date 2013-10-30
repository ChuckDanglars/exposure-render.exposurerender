
#include "guisocket.h"

QGuiSocket::QGuiSocket(int SocketDescriptor, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("compositor.ini", QSettings::IniFormat)
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
	
}