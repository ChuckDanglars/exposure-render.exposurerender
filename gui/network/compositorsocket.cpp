
#include "compositorsocket.h"

QCompositorSocket::QCompositorSocket(QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("gui.ini", QSettings::IniFormat),
	Estimate()
{
}

QCompositorSocket::~QCompositorSocket()
{
}

void QCompositorSocket::OnReceiveData(const QString& Action, QByteArray& Data)
{
	// qDebug() << Action.lower();

	if (Action == "ESTIMATE")
	{
		this->Estimate.FromByteArray(Data);
	}
}