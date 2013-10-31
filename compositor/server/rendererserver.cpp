
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

	this->Estimate.GetBuffer() = ((QRendererSocket*)this->Connections[0])->Estimate.GetBuffer();

	QByteArray Data;

	this->Estimate.ToByteArray(Data);

	this->GuiServer->SendDataToAll("ESTIMATE", Data);
}

