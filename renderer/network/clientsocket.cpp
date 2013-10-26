
#include "clientsocket.h"
#include "core\renderthread.h"

#include <QImage>
#include <QBuffer>

#include <time.h>

QClientSocket::QClientSocket(QRenderer* Renderer, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("renderer.ini", QSettings::IniFormat),
	Renderer(Renderer),
	ImageTimer(),
	RenderStatsTimer(),
	AvgEncodeSpeed(),
	GpuJpegEncoder()
{
	connect(&this->ImageTimer, SIGNAL(timeout()), this, SLOT(OnSendImage()));
	connect(&this->RenderStatsTimer, SIGNAL(timeout()), this, SLOT(OnSendRenderStats()));

	this->GpuJpegEncoder.Initialize(640, 480, 3);

	this->Renderer->Start();

	this->ImageTimer.start(1000.0f / this->Settings.value("network/sendimagefps ", 30).toInt());
	this->RenderStatsTimer.start(1000.0f / this->Settings.value("network/sendrenderstatsfps ", 20).toInt());
};

void QClientSocket::OnData(const QString& Action, QDataStream& DataStream)
{
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

		this->Renderer->Camera.SetPos(Vec3f(Position));
		this->Renderer->Camera.SetTarget(Vec3f(FocalPoint));
		this->Renderer->Camera.SetUp(Vec3f(ViewUp));

		this->Renderer->Camera.GetFilm().Restart();
	}

	if (Action == "IMAGE_SIZE")
	{
		unsigned int ImageSize[2] = { 0 };

		DataStream >> ImageSize[0];
		DataStream >> ImageSize[1];
			
		qDebug() << "Image size:" << ImageSize[0] << "x" << ImageSize[1];

		this->Renderer->Camera.GetFilm().Resize(Vec2i(ImageSize[0], ImageSize[1]));

		this->GpuJpegEncoder.Initialize(ImageSize[0], ImageSize[1], 3);

		this->Renderer->Start();
	}
}

void QClientSocket::OnSendImage()
{
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	QByteArray ImageBytes;
	
	this->GpuJpegEncoder.Encode((unsigned char*)this->Renderer->Camera.GetFilm().GetHostRunningEstimate().GetData());

	int CompressedImageSize = 0;
	unsigned char* CompressedImage = this->GpuJpegEncoder.GetCompressedImage(CompressedImageSize);

	ImageBytes.setRawData((char*)CompressedImage, CompressedImageSize);
	
	DataStream << (quint32)0;
	DataStream << QString("IMAGE");
	DataStream << this->AvgEncodeSpeed.GetAverageValue();
	DataStream << ImageBytes;

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));
	
	this->write(ByteArray);
	this->flush();
}

void QClientSocket::OnSendRenderStats()
{
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << (quint32)0;
	DataStream << QString("RENDER_STATS");
	DataStream << this->Renderer->AvgFps.GetAverageValue();

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));
	
	this->write(ByteArray);
	this->flush();
}