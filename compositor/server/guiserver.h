
#pragma once

#include <QTcpServer>
#include <QSettings>

class QGuiSocket;

class QGuiServer : public QTcpServer
{
	Q_OBJECT
public:
	explicit QGuiServer(QObject* Parent = 0);
	
	void Start();

protected:
	void incomingConnection(int SocketDescriptor);

signals:

public slots:

private:
	QSettings						Settings;
	QList<QGuiSocket*>			Connections;
};