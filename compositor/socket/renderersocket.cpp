
#include "renderersocket.h"
#include "server\guiserver.h"

#include <time.h>

#include <QImage>

QRendererSocket::QRendererSocket(int SocketDescriptor, QGuiServer* GuiServer, QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	GuiServer(GuiServer),
	Estimate()
{
	if (!this->setSocketDescriptor(SocketDescriptor))
		return;

	qDebug() << SocketDescriptor << "renderer connected";
}

QRendererSocket::~QRendererSocket()
{
}

void QRendererSocket::OnReceiveData(const QString& Action, QByteArray& Data)
{
	// qDebug() << Action.lower();

	if (Action == "ESTIMATE")
	{
		this->Estimate.FromByteArray(Data);
		this->GuiServer->SendDataToAll(Action, Data);
	}
}