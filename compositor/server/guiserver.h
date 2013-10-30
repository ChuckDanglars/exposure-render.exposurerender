
#pragma once

#include "utilities\network\baseserver.h"

#include <QSettings>

class QRendererServer;

class QGuiServer : public QBaseServer
{
	Q_OBJECT
public:
	explicit QGuiServer(QRendererServer* RendererServer, QObject* Parent = 0);
	
protected:
	void OnNewConnection(const int& SocketDescriptor);
	void OnStarted();

private:
	QSettings					Settings;
	QRendererServer*			RendererServer;
};