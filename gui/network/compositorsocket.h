#pragma once

#include "utilities\network\basesocket.h"
#include "utilities\general\estimate.h"

#include <QSettings>

class QCompositorSocket : public QBaseSocket
{
    Q_OBJECT

public:
    QCompositorSocket(QObject* Parent = 0);
	virtual ~QCompositorSocket();

protected:
	void OnReceiveData(const QString& Action, QByteArray& Data);

private:
	QSettings		Settings;
	QEstimate		Estimate;

friend class QGuiWindow;
};