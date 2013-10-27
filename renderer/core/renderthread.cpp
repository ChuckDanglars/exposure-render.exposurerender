
#include "core\renderthread.h"
#include "core\render.cuh"

#include <QSettings>
#include <QBuffer>
#include <QByteArray>
#include <QTimer>
#include <QDebug>
#include <QFile>

#include <time.h>

QRenderer::QRenderer(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Settings("renderer.ini", QSettings::IniFormat),
	RenderTimer(),
	AvgFps(),
	Renderer()
{
	connect(&this->RenderTimer, SIGNAL(timeout()), this, SLOT(OnRender()));

	this->Renderer.Camera.GetFilm().Block.x	= Settings.value("cuda/blockwidth", 8).toInt();
	this->Renderer.Camera.GetFilm().Block.y	= Settings.value("cuda/blockheight", 8).toInt();

	QFile File("C:\\workspaces\\manix.raw");

	QByteArray Voxels = File.readAll();

	this->Renderer.Volume.Voxels.Resize(Vec3i(256, 230, 256));
	this->Renderer.Volume.Voxels.FromHost((short*)Voxels.data());
}

void QRenderer::Start()
{
	this->RenderTimer.start(Settings.value("rendering/targetfps", 60).toInt());
}

void QRenderer::OnRender()
{
	this->Renderer.Camera.SetApertureSize(0.005f);
	this->Renderer.Camera.SetFocalDistance(0.1f);

	this->Renderer.Camera.Update();
	
	const clock_t Begin = clock();
	
	ExposureRender::Render(&this->Renderer);

	const clock_t End = clock();

	AvgFps.PushValue(1000.0f / (End - Begin));
}