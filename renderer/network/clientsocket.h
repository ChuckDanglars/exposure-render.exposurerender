
#include <QTimer>
#include <QSettings>

#include "basesocket.h"
#include "hysteresis.h"
#include "gpujpeg.h"

class QRenderer;

class QClientSocket : public QBaseSocket
{
    Q_OBJECT

public:
	QClientSocket(QRenderer* Renderer, QObject* Parent = 0);

	void OnData(const QString& Action, QDataStream& DataStream);

public slots:
	void OnSendImage();
	void OnSendRenderStats();

public:
	QSettings 			Settings;
	QRenderer*			Renderer;
	QTimer				ImageTimer;
	QTimer				RenderStatsTimer;
	QHysteresis			AvgEncodeSpeed;
	QGpuJpegEncoder		GpuJpegEncoder;
};