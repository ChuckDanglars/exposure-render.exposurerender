
#include "network\baseserver.h"

QBaseServer::QBaseServer(const QString& Name, QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Name(Name),
	ListenPort(0)
{
}

void QBaseServer::Start()
{
	qDebug() << "Starting" << this->Name.toLower() << "server" << this->ListenPort;

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
	// qDebug() << this->Name << "send" << Action.lower() << "to all";

	for (int s = 0; s < this->Connections.size(); s++)
		this->Connections[s]->SendData(Action, Data);
}

void QBaseServer::OnNewConnection(const int& SocketDescriptor)
{
	qDebug() << "Not implemented";
}

void QBaseServer::OnStarted()
{
	qDebug() << "Not implemented";
}