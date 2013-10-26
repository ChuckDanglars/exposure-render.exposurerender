
#include "core\renderthread.h"
#include "core\render.cuh"

#include <QSettings>
#include <QBuffer>
#include <QByteArray>
#include <QTimer>
#include <QDebug>

#include <time.h>

QRenderer::QRenderer(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Estimate(),
	Settings("renderer.ini", QSettings::IniFormat),
	RenderTimer(),
	AvgFps(),
	Renderer()
{
	this->Estimate.Resize(Vec2i(1024, 768));

	this->Position[0]		= 0.0f;
	this->Position[1]		= 1.0f;
	this->Position[2]		= 0.0f;
	this->FocalPoint[0]		= 0.0f;
	this->FocalPoint[1]		= 0.0f;
	this->FocalPoint[2]		= 0.0f;
	this->ViewUp[0]			= 0.0f;
	this->ViewUp[1]			= 0.0f;
	this->ViewUp[2]			= 1.0f;

	connect(&this->RenderTimer, SIGNAL(timeout()), this, SLOT(OnRender()));
}

void QRenderer::Start()
{
	this->RenderTimer.start(Settings.value("rendering/targetfps", 60).toInt());
}

void QRenderer::OnRender()
{
	const clock_t Begin = clock();

	// ExposureRender::Render(this->Position, this->FocalPoint, this->ViewUp, this->Estimate.Width(), this->Estimate.Height(), (unsigned char*)this->Estimate.GetData());

	const clock_t End = clock();

	AvgFps.PushValue(1000.0f / (End - Begin));
}