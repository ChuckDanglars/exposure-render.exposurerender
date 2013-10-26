
#pragma once

#include <QTcpServer>
#include <QDebug>
#include <QTimer>
#include <QList>
#include <QSettings>

#include "utilities\hysteresis.h"
#include "renderer\buffer\host\hostbuffer2d.h"
#include "renderer\color\color.h"

class QClientSocket;

using namespace ExposureRender;

class QServer : public QTcpServer
{
	Q_OBJECT
public:
	explicit QServer(QObject* Parent = 0);
	
	void Start();

protected:
	void incomingConnection(int SocketDescriptor);

signals:
	void newThreadedSocket(QClientSocket*);
	void UpdateEstimateIntegration(const float&);
	void CameraUpdate(float*, float*, float*);

public slots:
	void OnCombineEstimates();
	void OnCameraUpdate(float* Position, float* FocalPoint, float* ViewUp);

private:
	QSettings						Settings;
	QTimer							Timer;
	QList<QClientSocket*>			Connections;
	QHysteresis						AvgCombineTime;
	HostBuffer2D<ColorRGBuc>		Estimate;
	int								ImageSize[2];

	friend class QCompositorWindow;
	friend class QClientSocket;
};