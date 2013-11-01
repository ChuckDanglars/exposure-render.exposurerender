#pragma once

#include <QTcpSocket>
#include <QDataStream>

class QSocketData : public QObject
{
    Q_OBJECT

public:
    QSocketData(QObject* Parent = 0);
	virtual ~QSocketData();

	virtual void Receive(QByteArray& Data);
	virtual void Send(QByteArray& Data);

private:
};