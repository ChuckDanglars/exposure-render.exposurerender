
#include "server\rendererserver.h"
#include "clientsocket.h"
#include "combine.cuh"

#include <QtGui>

#include <time.h>

QServer::QServer(QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Timer(),
	Settings("compositor.ini", QSettings::IniFormat),
	Connections(),
	AvgCombineTime(),
	Estimate()
{
	this->ImageSize[0]	= this->Settings.value("rendering/imagewidth", 1024).toUInt();
	this->ImageSize[1]	= this->Settings.value("rendering/imageheight", 768).toUInt();

	connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnCombineEstimates()));
}

void QServer::Start()
{
	const int Port = this->Settings.value("network/rendererport", 6001).toInt();

	qDebug() << "Starting Exposure Render server";

	if (!this->listen(QHostAddress::Any, Port))
	{
		qDebug() << "Could not start server";
	}
	else
	{
		qDebug() << "Server listening to any ip on port" << Port;

		const int CombineFps = this->Settings.value("general/combinefps", 30).toFloat();

		qDebug() << "Estimate combination frequency:" << CombineFps << "fps";
		qDebug() << "Image dimensions:" << this->ImageSize[0] << "x" << this->ImageSize[1] << "pixels";

		this->Timer.start(1000.0f / CombineFps);

		this->Estimate.Resize(Vec2i(this->ImageSize[0], this->ImageSize[1]));
	}
}

void QServer::incomingConnection(int SocketDescriptor)
{
	qDebug() << SocketDescriptor << "connecting...";
	
	QClientSocket* ClientSocket = new QClientSocket(SocketDescriptor, this);
	
	this->Connections.append(ClientSocket);

	emit newThreadedSocket(ClientSocket);

	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	QByteArray ImageBytes;

	DataStream << (quint32)0;
	DataStream << QString("IMAGE_SIZE");
	DataStream << this->ImageSize[0];
	DataStream << this->ImageSize[1];

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));
	
	ClientSocket->write(ByteArray);
	ClientSocket->flush();
}

void QServer::OnCombineEstimates()
{
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
}

void QServer::OnCameraUpdate(float* Position, float* FocalPoint, float* ViewUp)
{
	for (int c = 0; c <this->Connections.size(); c++)
		this->Connections[c]->SendCamera(Position, FocalPoint, ViewUp);
}
