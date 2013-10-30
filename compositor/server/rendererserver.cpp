
#include "server\rendererserver.h"
#include "socket\renderersocket.h"
#include "cuda\combine.cuh"

QRendererServer::QRendererServer(QObject* Parent /*= 0*/) :
	QBaseServer("Renderer", Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	Timer(),
	Estimate()
{
	this->ListenPort = Settings.value("network/rendererport", 6001).toInt();

//	connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnCombineEstimates()));
}

void QRendererServer::OnNewConnection(const int& SocketDescriptor)
{
	QRendererSocket* RendererSocket = new QRendererSocket(SocketDescriptor, this);
	this->Connections.append(RendererSocket);
}

void QRendererServer::OnStarted()
{
	const int CombineFps = this->Settings.value("general/combinefps", 30).toFloat();

	this->Timer.start(1000.0f / CombineFps);
}