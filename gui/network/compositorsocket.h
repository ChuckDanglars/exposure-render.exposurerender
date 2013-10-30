#pragma once

#include "utilities\network\basesocket.h"

#include <QSettings>

class QCompositorSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QCompositorSocket(QObject* Parent = 0);
	virtual ~QCompositorSocket();

private:
	QSettings		Settings;
};