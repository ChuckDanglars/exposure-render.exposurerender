#pragma once

#include <QObject>

#include "gpujpeg/gpujpeg.h"

struct gpujpeg_encoder;

class QGpuJpegEncoder : public QObject
{
    Q_OBJECT

public:
	QGpuJpegEncoder(QObject* Parent = 0);
	virtual ~QGpuJpegEncoder();
	
	void Encode(unsigned char* Image);
	unsigned char* GetCompressedImage(int& Size);

	void Initialize(const unsigned int& Width, const unsigned int& Height, const unsigned int& NoComponentsPerPixel);

private:
	struct gpujpeg_encoder*			Encoder;
	bool							Initialized;
	unsigned char*					CompressedImage;
	int								CompressedSize;
};

class QGpuJpegDecoder : public QObject
{
    Q_OBJECT

public:
	QGpuJpegDecoder(QObject* Parent = 0);
	virtual ~QGpuJpegDecoder();
	
	void Decode(unsigned char* CompressedImage, const int& CompressedImageSize, int& Width, int& Height, int& NoBytes);
	unsigned char* GetImage(int& Size);

private:
	void Initialize();

private:
	struct gpujpeg_decoder*			Decoder;
	struct gpujpeg_decoder_output	DecoderOutput;
	bool							Initialized;
	uint8_t*						Image;
	int								ImageSize;
};