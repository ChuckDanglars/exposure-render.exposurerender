
#include "camerasocketdata.h"

#include <QFile>
#include <QFileInfo>

QCameraSocketData::QCameraSocketData(QObject* Parent /*= 0*/) :
	QSocketData(Parent)
{
}

QCameraSocketData::~QCameraSocketData()
{
}

void QCameraSocketData::Receive(QByteArray& Data)
{
	
	
	QDataStream DataStream(&Data, QIODevice::ReadOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	QByteArray Voxels;

	//DataStream.writeBytes((char*)&this->Camera, sizeof(this->Camera));

	/*
	DataStream >> this->FileName;
		
	DataStream >> this->Resolution[0];
	DataStream >> this->Resolution[1];
	DataStream >> this->Resolution[2];

	DataStream >> this->Spacing[0];
	DataStream >> this->Spacing[1];
	DataStream >> this->Spacing[2];
		
	DataStream >> Voxels;
	DataStream >> this->NoBytes;

	memcpy(this->Voxels, Voxels.data(), this->NoBytes);
	*/
}

void QCameraSocketData::Send(QBaseSocket* Socket, QByteArray& Data)
{
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	/*
	DataStream << FileInfo.fileName();
		
	DataStream << this->Resolution[0];
	DataStream << this->Resolution[1];
	DataStream << this->Resolution[2];

	DataStream << this->Spacing[0];
	DataStream << this->Spacing[1];
	DataStream << this->Spacing[2];
		
	DataStream << Voxels;
	DataStream << this->NoBytes;
	*/

	QSocketData::Send(Socket, ByteArray);
}
