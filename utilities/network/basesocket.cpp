
#include "basesocket.h"

#include <QFile>
#include <QApplication>

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

		QByteArray ByteArray(this->BlockSize);

		DataStream.readRawData(ByteArray.data(), this->BlockSize);

		QDataStream ActionDataStream(&ByteArray, QIODevice::ReadOnly);

		ActionDataStream >> Action;

		QByteArray ActionData = ActionDataStream.device()->readAll();

		this->OnReceiveData(Action, ActionData);

		this->BlockSize = 0;
	}
}

void QBaseSocket::OnReceiveData(const QString& Action, QByteArray& ByteArray)
{
	qDebug() << "Not implemented";
}

void QBaseSocket::SendData(const QString& Action, QByteArray& Data)
{
	qDebug() << "Sending" << Action.lower();

	QByteArray ByteArray;

	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << quint32(0);

	DataStream << Action;

	ByteArray.append(Data);

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));

    this->write(ByteArray);
	this->flush();
}

void QBaseSocket::SaveResource(QByteArray& ByteArray)
{
	QDataStream DataStream(&ByteArray, QIODevice::ReadWrite);
	DataStream.setVersion(QDataStream::Qt_4_0);

	QString FileName;
	QByteArray Data;

	DataStream >> FileName;
	DataStream >> Data;

	qDebug() << "Saving resource" << FileName << Data.count() << "bytes";

	QFile File(QApplication::applicationDirPath() + "//resources//" + FileName);
	
	if (File.open(QIODevice::WriteOnly))
	{
		File.write(Data);
		File.close();

		qDebug() << "Saved" << File.fileName();
	}
	else
	{
		qDebug() << "Unable to save" << File.fileName();
	}
}