
#include "network\baseserver.h"

QBaseServer::QBaseServer(const QString& Name, QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Name(Name),
	ListenPort(0)
{
}

void QBaseServer::Start()
{
	qDebug() << "Starting" << this->Name.toLower() << "server";

	if (!this->listen(QHostAddress::Any, this->ListenPort))
	{
		qDebug() << "Could not start server";
	}
	else
	{
		qDebug() << "Server listening to any ip on port" << this->ListenPort;
	}

	this->OnStarted();
}

void QBaseServer::SendDataToAll(const QString& Action, QByteArray& Data)
{
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << quint32(0);

	DataStream << Action;
	DataStream << Data;

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));

	for (int s = 0; s < this->Connections.size(); s++)
	{
		this->Connections[s]->write(ByteArray);
		this->Connections[s]->flush();
	}
}

void QBaseServer::OnNewConnection(const int& SocketDescriptor)
{
	qDebug() << "Not implemented";
}

void QBaseServer::OnStarted()
{
	qDebug() << "Not implemented";
}