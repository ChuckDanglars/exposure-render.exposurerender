#pragma once

#include <QObject>
#include <QSettings>
#include <QTimer>

#include "hysteresis.h"
#include "buffer\buffers.h"
#include "color\color.h"
#include "core\renderer.h"

using namespace ExposureRender;

class QRenderer : public QObject
{
    Q_OBJECT

public:
	QRenderer(QObject* Parent = 0);
	virtual ~QRenderer() {};

	void Start();

public slots:
	void OnRender();

public:
	HostBuffer2D<ColorRGBuc>	Estimate;
	QSettings 					Settings;
	QTimer						RenderTimer;
	QHysteresis					AvgFps;
	float						Position[3];
	float						FocalPoint[3];
	float						ViewUp[3];
	ExposureRender::Renderer	Renderer;
};