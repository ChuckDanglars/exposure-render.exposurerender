#pragma once

#include "renderer\buffer\buffers.h"
#include "renderer\color\colorrgbuc.h"
#include "utilities\general\hysteresis.h"
#include "utilities\gpujpeg\gpujpeg.h"
#include "utilities\network\basesocket.h"

#include <QDebug>
#include <QSettings>

using namespace ExposureRender;

class QGuiServer;

class QRendererSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QRendererSocket(int SocketDescriptor, QGuiServer* GuiServer, QObject* Parent = 0);
	virtual ~QRendererSocket();

	void OnReceiveData(const QString& Action, QByteArray& Data);

private:
	QSettings						Settings;
	QGuiServer*						GuiServer;
	QGpuJpegDecoder					GpuJpegDecoder;
	HostBuffer2D<ColorRGBuc>		Estimate;

friend class QServer;
friend class QConnectionItem;
friend class QMainDialog;
};