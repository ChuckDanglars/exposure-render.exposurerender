#pragma once

#include "utilities\network\basesocket.h"

#include <QSettings>

#include "buffer\buffers.h"
#include "color\color.h"

using namespace ExposureRender;

class QCompositorSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QCompositorSocket(QObject* Parent = 0);
	virtual ~QCompositorSocket();

protected:
	void OnReceiveData(const QString& Action, QByteArray& Data);

private:
	QSettings					Settings;
	HostBuffer2D<ColorRGBuc>	Estimate;

friend class QGuiWindow;
};