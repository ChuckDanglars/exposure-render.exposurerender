
#include "server\rendererserver.h"
#include "socket\renderersocket.h"
#include "cuda\combine.cuh"

QRendererServer::QRendererServer(QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Timer(),
	Settings("compositor.ini", QSettings::IniFormat),
	Connections(),
	AvgCombineTime(),
	Estimate()
{
	connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnCombineEstimates()));
}

void QRendererServer::Start()
{
	const int Port = this->Settings.value("network/rendererport", 6001).toInt();

	qDebug() << "Starting Exposure Render compositor server";

	if (!this->listen(QHostAddress::Any, Port))
	{
		qDebug() << "Could not start server";
	}
	else
	{
		qDebug() << "Renderer server listening to any ip on port" << Port;

		const int CombineFps = this->Settings.value("general/combinefps", 30).toFloat();

		this->Timer.start(1000.0f / CombineFps);
	}
}

void QRendererServer::incomingConnection(int SocketDescriptor)
{
	qDebug() << SocketDescriptor << "connecting...";
	
	QRendererSocket* RendererSocket = new QRendererSocket(SocketDescriptor, this);
	
	this->Connections.append(RendererSocket);
}

void QRendererServer::OnCombineEstimates()
{
	/*
	if (this->Connections.size() == 0)
		return;

	unsigned char* Estimates[20];
	int NoEstimates = 0;

	for (int c = 0; c < this->Connections.size(); c++)
	{
		if (this->Connections[c]->state() == QAbstractSocket::ConnectedState)
		{
			Estimates[c] = (unsigned char*)this->Connections[c]->Estimate.GetData();
			NoEstimates++;
		}
	}

	this->AvgCombineTime.PushValue(ExposureRender::Combine(this->Estimate.Width(), this->Estimate.Height(), Estimates, NoEstimates, (unsigned char*)this->Estimate.GetData()));

	// qDebug() << this->AvgCombineTime.GetAverageValue();
	*/
}