
#include "server\guiserver.h"
#include "server\guisocket.h"

#include <QDebug>

#include <QtGui>

QGuiServer::QGuiServer(QObject* Parent /*= 0*/) :
	QTcpServer(Parent),
	Settings("compositor.ini", QSettings::IniFormat),
	Connections()
{
	connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnCombineEstimates()));
}

void QGuiServer::Start()
{
	const int Port = this->Settings.value("network/guiport", 6000).toInt();

	qDebug() << "Starting Exposure Render server";

	if (!this->listen(QHostAddress::Any, Port))
	{
		qDebug() << "Could not start server";
	}
	else
	{
		qDebug() << "Server listening to any ip on port" << Port;
	}
}

void QGuiServer::incomingConnection(int SocketDescriptor)
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
