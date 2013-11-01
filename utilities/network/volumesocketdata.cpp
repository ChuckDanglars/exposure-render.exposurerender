
#include "volumesocketdata.h"

#include <QFile>
#include <QFileInfo>

QVolumeSocketData::QVolumeSocketData(QObject* Parent /*= 0*/) :
	QSocketData(Parent),
	Voxels(0),
	NoBytes(0)
{
}

QVolumeSocketData::~QVolumeSocketData()
{
}

void QVolumeSocketData::Receive(QByteArray& Data)
{
	QByteArray Voxels;
	
	QDataStream DataStream(&Data, QIODevice::ReadOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

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
}

void QVolumeSocketData::Send(QBaseSocket* Socket, QByteArray& Data)
{
	QByteArray Voxels((char*)this->Voxels, this->NoBytes);
	
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	QFileInfo FileInfo(this->FileName);

	DataStream << FileInfo.fileName();
		
	DataStream << this->Resolution[0];
	DataStream << this->Resolution[1];
	DataStream << this->Resolution[2];

	DataStream << this->Spacing[0];
	DataStream << this->Spacing[1];
	DataStream << this->Spacing[2];
		
	DataStream << Voxels;
	DataStream << this->NoBytes;

	QSocketData::Send(Socket, ByteArray);
}

void QVolumeSocketData::FromDisk(const QString& FileName)
{
	this->FileName = FileName;

	//QString FileName = "C://workspaces//manix.raw";

	QFile File(FileName);

	if (File.open(QIODevice::ReadOnly))
	{
		this->Resolution[0]	= 256;
		this->Resolution[1]	= 230;
		this->Resolution[2]	= 256;

		this->Spacing[0]	= 1.0f;
		this->Spacing[1]	= 1.0f;
		this->Spacing[2]	= 1.0f;

		this->NoBytes	= File.size();
		this->Voxels	= (short*)malloc(this->NoBytes);

		memset(this->Voxels, 0, this->NoBytes);

		File.read((char*)this->Voxels, this->NoBytes);

		qDebug() << "Read volume from disk";
	}
	else
	{
		qDebug() << "Unable to read volume from disk";
	}
}
