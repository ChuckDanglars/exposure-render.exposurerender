#pragma once

#include "socketdata.h"

#include <QString>

class QCameraSocketData : public QSocketData
{
    Q_OBJECT

public:
    QCameraSocketData(QObject* Parent = 0);
	virtual ~QCameraSocketData();

	virtual void Receive(QByteArray& Data);
	virtual void Send(QBaseSocket* Socket, QByteArray& Data);

private:
};