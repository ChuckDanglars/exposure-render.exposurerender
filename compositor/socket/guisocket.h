#pragma once

#include "renderer\buffer\buffers.h"
#include "renderer\color\colorrgbuc.h"
#include "utilities\hysteresis.h"
#include "utilities\gpujpeg.h"
#include "utilities\basesocket.h"

#include <QDebug>
#include <QSettings>

using namespace ExposureRender;

class QGuiSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QGuiSocket(int SocketDescriptor, QObject* Parent = 0);
	virtual ~QGuiSocket();

	void OnData(const QString& Action, QDataStream& DataStream);

signals:

private:
	QSettings						Settings;

friend class QServer;
};