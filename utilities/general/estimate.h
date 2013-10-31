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
	
	bool Encode(QByteArray& Data);
	bool Decode(QByteArray& Data);
	bool ToByteArray(QByteArray& Data);
	bool FromByteArray(QByteArray& Data);

	HostBuffer2D<ColorRGBuc>& GetBuffer() { return this->Buffer; }

private:
	HostBuffer2D<ColorRGBuc>	Buffer;
	QGpuJpegEncoder				GpuJpegEncoder;
	QGpuJpegDecoder				GpuJpegDecoder;
};