#pragma once

#include "basesocket.h"

class QBaseSocket;

class QSocketData : public QObject
{
    Q_OBJECT

public:
    QSocketData(QObject* Parent = 0);
	virtual ~QSocketData();

	virtual void Receive(QByteArray& Data);
	virtual void Send(QBaseSocket* Socket, QByteArray& Data);
};