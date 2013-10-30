
#pragma once

#include "basesocket.h"

#include <QTcpServer>

class QBaseServer : public QTcpServer
{
	Q_OBJECT

public:
	QBaseServer(const QString& Name, QObject* Parent = 0);

	void Start();

	void SendDataToAll(const QString& Action, QByteArray& Data);

protected:
	void incomingConnection(int SocketDescriptor)
	{
		qDebug() << SocketDescriptor << "connecting...";
	
		this->OnNewConnection(SocketDescriptor);
	}

	virtual void OnNewConnection(const int& SocketDescriptor);
	virtual void OnStarted();

protected:
	QString					Name;
	int						ListenPort;
	QList<QBaseSocket*>		Connections;
};