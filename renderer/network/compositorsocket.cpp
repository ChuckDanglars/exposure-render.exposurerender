
#include "compositorsocket.h"
#include "core\renderthread.h"

#include <QImage>
#include <QBuffer>

#include <time.h>

QCompositorSocket::QCompositorSocket(QRenderer* Renderer, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("renderer.ini", QSettings::IniFormat),
	Renderer(Renderer),
	ImageTimer(),
	RenderStatsTimer(),
	Estimate()
{
	connect(&this->ImageTimer, SIGNAL(timeout()), this, SLOT(OnSendImage()));
	connect(&this->RenderStatsTimer, SIGNAL(timeout()), this, SLOT(OnSendRenderStats()));

	this->Renderer->Start();

	this->ImageTimer.start(1000.0f / this->Settings.value("network/sendimagefps ", 30).toInt());
	this->RenderStatsTimer.start(1000.0f / this->Settings.value("network/sendrenderstatsfps ", 20).toInt());
};

void QCompositorSocket::OnReceiveData(const QString& Action, QByteArray& ByteArray)
{
	if (Action == "VOLUME" || Action == "BITMAP")
	{
		qDebug() << "Received" << Action.toLower();

		this->SaveResource(ByteArray);
	}

	/*
	if (Action == "CAMERA")
	{
		float Position[3], FocalPoint[3], ViewUp[3];

		DataStream >> Position[0];
		DataStream >> Position[1];
		DataStream >> Position[2];

		DataStream >> FocalPoint[0];
		DataStream >> FocalPoint[1];
		DataStream >> FocalPoint[2];

		DataStream >> ViewUp[0];
		DataStream >> ViewUp[1];
		DataStream >> ViewUp[2];

		this->Renderer->Renderer.Camera.SetPos(Vec3f(Position));
		this->Renderer->Renderer.Camera.SetTarget(Vec3f(FocalPoint));
		this->Renderer->Renderer.Camera.SetUp(Vec3f(ViewUp));
		
		this->Renderer->Renderer.Camera.GetFilm().Restart();
	}

	if (Action == "IMAGE_SIZE")
	{
		unsigned int ImageSize[2] = { 0 };

		DataStream >> ImageSize[0];
		DataStream >> ImageSize[1];
			
		qDebug() << "Image size:" << ImageSize[0] << "x" << ImageSize[1];

		this->Renderer->Renderer.Camera.GetFilm().Resize(Vec2i(ImageSize[0], ImageSize[1]));

		this->GpuJpegEncoder.Initialize(ImageSize[0], ImageSize[1], 3);

		this->Renderer->Start();
	}
	*/
}

void QCompositorSocket::OnSendImage()
{
	this->Estimate.GetBuffer() = this->Renderer->Renderer.Camera.GetFilm().GetHostRunningEstimate();

	QByteArray CompressedImage;

	this->Estimate.ToByteArray(CompressedImage);

	this->SendData("ESTIMATE", CompressedImage);
}

void QCompositorSocket::OnSendRenderStats()
{
}