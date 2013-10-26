
#include "basesocket.h"

QBaseSocket::QBaseSocket(QObject* Parent /*= 0*/) :
	QTcpSocket(Parent),
	BlockSize(0)
{
	connect(this, SIGNAL(readyRead()), this, SLOT(OnReadyRead()), Qt::DirectConnection);
}

QBaseSocket::~QBaseSocket()
{
}

void QBaseSocket::OnReadyRead()
{
	QDataStream DataStream(this);

    DataStream.setVersion(QDataStream::Qt_4_0);

	if (this->bytesAvailable() < this->BlockSize)
		return;

	while (this->bytesAvailable() >=sizeof(quint32))
	{
		if (this->BlockSize == 0)
			DataStream >> this->BlockSize;

		if (this->bytesAvailable() < this->BlockSize)
			return;

		QString Action;

		DataStream >> Action;

		this->OnData(Action, DataStream);

		this->BlockSize = 0;
	}
}

void QBaseSocket::OnData(const QString& Action, QDataStream& DataStream)
{
	qDebug() << "Not implemented";
}