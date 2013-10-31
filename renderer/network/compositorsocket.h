
#include <QTimer>
#include <QSettings>

#include "utilities\network\basesocket.h"
#include "utilities\general\hysteresis.h"
#include "utilities\general\estimate.h"

class QRenderer;

class QCompositorSocket : public QBaseSocket
{
    Q_OBJECT

public:
	QCompositorSocket(QRenderer* Renderer, QObject* Parent = 0);

	void OnReceiveData(const QString& Action, QByteArray& Data);

public slots:
	void OnSendImage();
	void OnSendRenderStats();

public:
	QSettings 			Settings;
	QRenderer*			Renderer;
	QTimer				ImageTimer;
	QTimer				RenderStatsTimer;
	QEstimate			Estimate;
};