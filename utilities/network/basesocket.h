#pragma once

#include <QTcpSocket>
#include <QDebug>

class QBaseSocket : public QTcpSocket
{
    Q_OBJECT

public:
    QBaseSocket(QObject* Parent = 0);
	virtual ~QBaseSocket();

public:
	virtual void OnReceiveData(const QString& Action, QDataStream& DataStream);
	
	void SendData(const QString& Action, QByteArray& Data);

public slots:
	void OnReadyRead();

private:
	quint32		BlockSize;
};