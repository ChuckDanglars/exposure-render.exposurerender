#pragma once

#include <QTcpSocket>
#include <QDataStream>

class QBaseSocket : public QTcpSocket
{
    Q_OBJECT

public:
    QBaseSocket(QObject* Parent = 0);
	virtual ~QBaseSocket();

public:
	virtual void OnReceiveData(const QString& Action, QByteArray& ByteArray);
	
	void SendData(const QString& Action, QByteArray& ByteArray);

protected:
	void SaveResource(QByteArray& ByteArray);

public slots:
	void OnReadyRead();

private:
	quint32		BlockSize;
};