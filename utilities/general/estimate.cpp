
#include "estimate.h"

#include <QDataStream>
#include <QDebug>

QEstimate::QEstimate(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Buffer(),
	GpuJpegEncoder(),
	GpuJpegDecoder()
{
}

void QEstimate::Encode(QByteArray& Data)
{
	if (this->Buffer.GetResolution().CumulativeProduct() == 0)
	{
		qDebug() << "Unable to encode 0 pixels";
		return;
	}

	this->GpuJpegEncoder.Initialize(this->Buffer.Width(), this->Buffer.Height(), 3);
	this->GpuJpegEncoder.Encode((unsigned char*)this->Buffer.GetData());

	int CompressedImageSize = 0;
	unsigned char* CompressedImage = this->GpuJpegEncoder.GetCompressedImage(CompressedImageSize);

	Data.setRawData((char*)CompressedImage, CompressedImageSize);
}

void QEstimate::Decode(QByteArray& Data)
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

	this->Buffer.Resize(Vec2i(Width, Height));
	memcpy(this->Buffer.GetData(), ImageData, NoBytes);	
}

void QEstimate::ToByteArray(QByteArray& Data)
{
	QByteArray EncodedImage;

	this->Encode(EncodedImage);

	QDataStream DataStream(&Data, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << this->Buffer.Width();
	DataStream << this->Buffer.Height();
	DataStream << EncodedImage;
}

void QEstimate::FromByteArray(QByteArray& Data)
{
	this->Decode(Data);
}
