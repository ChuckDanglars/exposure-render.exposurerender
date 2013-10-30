#pragma once

#include "renderer\buffer\buffers.h"
#include "renderer\color\colorrgbuc.h"
#include "utilities\network\basesocket.h"

#include <QSettings>

using namespace ExposureRender;

class QGuiSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QGuiSocket(int SocketDescriptor, QObject* Parent = 0);
	virtual ~QGuiSocket();

	void OnReceiveData(const QString& Action, QDataStream& DataStream);
	// void SendData(const QString& Action, QDataStream& DataStream);

signals:

private:
	QSettings						Settings;

friend class QServer;
};