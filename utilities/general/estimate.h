#pragma once

#include "utilities\gpujpeg\gpujpeg.h"
#include "buffer\buffers.h"
#include "color\color.h"

#include <QObject>

using namespace ExposureRender;

class QEstimate : public QObject
{
    Q_OBJECT

public:
	QEstimate(QObject* Parent = 0);
	virtual ~QEstimate() {};
	
	void Encode(QByteArray& Data);
	void Decode(QByteArray& Data);
	void ToByteArray(QByteArray& Data);
	void FromByteArray(QByteArray& Data);

	HostBuffer2D<ColorRGBuc>& GetBuffer() { return this->Buffer; }

private:
	HostBuffer2D<ColorRGBuc>	Buffer;
	QGpuJpegEncoder				GpuJpegEncoder;
	QGpuJpegDecoder				GpuJpegDecoder;
};