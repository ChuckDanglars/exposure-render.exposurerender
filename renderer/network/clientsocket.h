
#include <QTimer>
#include <QSettings>

#include "utilities\network\basesocket.h"
#include "utilities\general\hysteresis.h"
#include "utilities\gpujpeg\gpujpeg.h"

class QRenderer;

class QClientSocket : public QBaseSocket
{
    Q_OBJECT

public:
	QClientSocket(QRenderer* Renderer, QObject* Parent = 0);

	void OnReceiveData(const QString& Action, QDataStream& DataStream);

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