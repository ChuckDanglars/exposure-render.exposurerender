
#pragma once

#include "utilities\network\baseserver.h"

#include <QSettings>
#include <QTimer>

#include "renderer\buffer\host\hostbuffer2d.h"
#include "renderer\color\color.h"

using namespace ExposureRender;

class QRendererServer : public QBaseServer
{
	Q_OBJECT
public:
	explicit QRendererServer(QObject* Parent = 0);
	
protected:
	void OnNewConnection(const int& SocketDescriptor);
	void OnStarted();

private:
	QSettings					Settings;
	QTimer						Timer;
	HostBuffer2D<ColorRGBuc>	Estimate;

	friend class QCompositorWindow;
};