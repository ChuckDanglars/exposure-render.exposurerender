#pragma once

#include "socketdata.h"

class QVolume : public QSocketData
{
    Q_OBJECT

public:
    QVolume(QObject* Parent = 0);
	virtual ~QVolume();

	virtual void Receive(QByteArray& Data);
	virtual void Send(QByteArray& Data);

private:
	int		Resolution[3];

};