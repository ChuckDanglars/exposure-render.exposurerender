#pragma once

#include "socketdata.h"

#include <QString>

class QVolumeSocketData : public QSocketData
{
    Q_OBJECT

public:
    QVolumeSocketData(QObject* Parent = 0);
	virtual ~QVolumeSocketData();

	virtual void Receive(QByteArray& Data);
	virtual void Send(QBaseSocket* Socket, QByteArray& Data);

	void FromDisk(const QString& FileName);

private:
	QString		FileName;
	int			Resolution[3];
	float		Spacing[3];
	short*		Voxels;
	int			NoBytes;
};