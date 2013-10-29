
#include "clientsocket.h"
#include "server.h"

#include <time.h>

#include <QImage>

QClientSocket::QClientSocket(int SocketDescriptor, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	AvgDecodeSpeed(),
	GpuJpegDecoder(),
	Estimate()
{
	if (!this->setSocketDescriptor(SocketDescriptor))
		return;

	qDebug() << SocketDescriptor << "client connected";

	this->ImageSize[0]	= this->Settings.value("rendering/imagewidth", 1024).toUInt();
	this->ImageSize[1]	= this->Settings.value("rendering/imageheight", 768).toUInt();

	this->Estimate.Resize(Vec2i(this->ImageSize[0], this->ImageSize[1]));
}

QClientSocket::~QClientSocket()
{
}

void QClientSocket::OnData(const QString& Action, QDataStream& DataStream)
{
	if (Action == "IMAGE")
	{
		QByteArray ImageBytes;

		float JpgEncodeTime = 0.0f;

		DataStream >> JpgEncodeTime;
		DataStream >> ImageBytes;
		
		int Width = 0, Height = 0, NoBytes = 0;

		GpuJpegDecoder.Decode((unsigned char*)ImageBytes.data(), ImageBytes.count(), Width, Height, NoBytes);
		
		unsigned char* ImageData = GpuJpegDecoder.GetImage(NoBytes);

		this->Estimate.Resize(Vec2i(Width, Height));
		memcpy(this->Estimate.GetData(), ImageData, NoBytes);

		emit UpdateJpgEncodeTime(JpgEncodeTime);
		emit UpdateJpgNoBytes(ImageBytes.count());
	}

	if (Action == "RENDER_STATS")
	{
		float Fps = 0.0f;
			
		DataStream >> Fps;

		emit UpdateFps(Fps);
	}
}

void QClientSocket::SendCamera(float* Position, float* FocalPoint, float* ViewUp)
{
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << quint32(0);

	DataStream << QString("CAMERA");

	DataStream << Position[0];
	DataStream << Position[1];
	DataStream << Position[2];

	DataStream << FocalPoint[0];
	DataStream << FocalPoint[1];
	DataStream << FocalPoint[2];

	DataStream << ViewUp[0];
	DataStream << ViewUp[1];
	DataStream << ViewUp[2];

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));

    this->write(ByteArray);
	this->flush();
}