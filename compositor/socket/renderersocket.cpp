
#include "renderersocket.h"
#include "server\guiserver.h"

#include <time.h>

#include <QImage>

QRendererSocket::QRendererSocket(int SocketDescriptor, QGuiServer* GuiServer, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	GuiServer(GuiServer),
	GpuJpegDecoder(),
	Estimate()
{
	if (!this->setSocketDescriptor(SocketDescriptor))
		return;

	qDebug() << SocketDescriptor << "renderer connected";
}

QRendererSocket::~QRendererSocket()
{
}

void QRendererSocket::OnReceiveData(const QString& Action, QByteArray& Data)
{
	qDebug() << Action.lower();

	if (Action == "ESTIMATE")
	{
		QDataStream DataStream(&Data, QIODevice::ReadOnly);
		DataStream.setVersion(QDataStream::Qt_4_0);

		int Width = 0, Height = 0, NoBytes = 0;

		QByteArray CompressedImageBytes;

		DataStream >> Width;
		DataStream >> Height;
		DataStream >> CompressedImageBytes;

		GpuJpegDecoder.Decode((unsigned char*)CompressedImageBytes.data(), CompressedImageBytes.count(), Width, Height, NoBytes);
		
		unsigned char* ImageData = GpuJpegDecoder.GetImage(NoBytes);

		this->Estimate.Resize(Vec2i(Width, Height));
		memcpy(this->Estimate.GetData(), ImageData, NoBytes);	
	}

	/*
	if (Action == "RENDER_STATS")
	{
		float Fps = 0.0f;
			
		DataStream >> Fps;

		emit UpdateFps(Fps);
	}
	*/
}

/*
void QRendererSocket::SendCamera(float* Position, float* FocalPoint, float* ViewUp)
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
*/