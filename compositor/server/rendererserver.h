
#pragma once

#include <QTcpServer>
#include <QDebug>
#include <QTimer>
#include <QList>
#include <QSettings>

#include "utilities\hysteresis.h"
#include "renderer\buffer\host\hostbuffer2d.h"
#include "renderer\color\color.h"

class QRendererSocket;

using namespace ExposureRender;

class QRendererServer : public QTcpServer
{
	Q_OBJECT
public:
	explicit QRendererServer(QObject* Parent = 0);
	
	void Start();

protected:
	void incomingConnection(int SocketDescriptor);

public slots:
	void OnCombineEstimates();

private:
	QSettings									Settings;
	QTimer										Timer;
	QList<QRendererSocket*>						Connections;
	QHysteresis									AvgCombineTime;
	ExposureRender::HostBuffer2D<ColorRGBuc>	Estimate;

	friend class QCompositorWindow;
};