
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
	if (Action == "ESTIMATE")
	{
		QDataStream DataStream(&Data, QIODevice::ReadOnly);
		DataStream.setVersion(QDataStream::Qt_4_0);

		int Width = 0, Height = 0;

		QByteArray ImageBytes;

		DataStream >> Width;
		DataStream >> Height;
		DataStream >> ImageBytes;

		this->Estimate.SetData(ImageBytes.data(), Width, Height);
	}
}