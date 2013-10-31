#pragma once

#include "utilities\network\basesocket.h"
#include "utilities\general\estimate.h"

#include <QDebug>
#include <QSettings>

class QGuiServer;

class QRendererSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QRendererSocket(int SocketDescriptor, QGuiServer* GuiServer, QObject* Parent = 0);
	virtual ~QRendererSocket();

	void OnReceiveData(const QString& Action, QByteArray& Data);

private:
	QSettings		Settings;
	QGuiServer*		GuiServer;
	QEstimate		Estimate;

friend class QRendererServer;
};