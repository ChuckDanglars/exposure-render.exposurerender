#pragma once

#include "renderer\buffer\buffers.h"
#include "renderer\color\colorrgbuc.h"
#include "utilities\hysteresis.h"
#include "utilities\gpujpeg.h"
#include "utilities\basesocket.h"

#include <QDebug>
#include <QSettings>

using namespace ExposureRender;

class QClientSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QClientSocket(int SocketDescriptor, QObject* Parent = 0);
	virtual ~QClientSocket();

	void OnReceiveData(const QString& Action, QDataStream& DataStream);

public:
	void SendCamera(float* Position, float* FocalPoint, float* ViewUp);

private:
	QSettings						Settings;
	QHysteresis						AvgDecodeSpeed;
	QGpuJpegDecoder					GpuJpegDecoder;
	HostBuffer2D<ColorRGBuc>		Estimate;
	int								ImageSize[2];

friend class QServer;
friend class QConnectionItem;
friend class QMainDialog;
};