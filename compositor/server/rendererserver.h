#pragma once

#include "utilities\network\baseserver.h"
#include "utilities\general\estimate.h"

#include <QSettings>
#include <QTimer>

class QGuiServer;

class QRendererServer : public QBaseServer
{
	Q_OBJECT
public:
	explicit QRendererServer(QObject* Parent = 0);
	
	QGuiServer*					GuiServer;

protected:
	void OnNewConnection(const int& SocketDescriptor);
	void OnStarted();

public slots:
	void OnCombineEstimates();

private:
	QSettings		Settings;
	QTimer			Timer;
	QEstimate		Estimate;

	friend class QCompositorWindow;
};