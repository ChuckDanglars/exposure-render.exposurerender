
#include "server\rendererserver.h"
#include "server\guiserver.h"
#include "socket\renderersocket.h"
#include "cuda\combine.cuh"

QRendererServer::QRendererServer(QObject* Parent /*= 0*/) :
	QBaseServer("Renderer", Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	GuiServer(0),
	Timer(),
	Estimate()
{
	this->ListenPort = Settings.value("network/rendererport", 6000).toInt();

	connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnCombineEstimates()));
}

void QRendererServer::OnNewConnection(const int& SocketDescriptor)
{
	QRendererSocket* RendererSocket = new QRendererSocket(SocketDescriptor, this->GuiServer, this);
	this->Connections.append(RendererSocket);
}

void QRendererServer::OnStarted()
{
	const int CombineFps = this->Settings.value("general/combinefps", 30).toFloat();

	this->Timer.start(1000.0f / CombineFps);
}

void QRendererServer::OnCombineEstimates()
{
	if (this->Connections.size() == 0)
		return;

	unsigned char* Estimates[20];
	int NoEstimates = 0;

	for (int c = 0; c < this->Connections.size(); c++)
	{
		if (this->Connections[c]->state() == QAbstractSocket::ConnectedState)
		{
			QRendererSocket* RendererSocket = (QRendererSocket*)this->Connections[c];

			if (!RendererSocket->Estimate.GetBuffer().IsEmpty())
			{
				Estimates[NoEstimates] = (unsigned char*)(RendererSocket->Estimate.GetBuffer().GetData());
				NoEstimates++;
			}
		}
	}
	
	this->Estimate.GetBuffer().Resize(Vec2i(640, 480));

	if (NoEstimates > 0)
		ExposureRender::Combine(this->Estimate.GetBuffer().Width(), this->Estimate.GetBuffer().Height(), Estimates, NoEstimates, (unsigned char*)this->Estimate.GetBuffer().GetData());
	
	QByteArray Data;

	if (this->Estimate.ToByteArray(Data))
		this->GuiServer->SendDataToAll("ESTIMATE", Data);
}

