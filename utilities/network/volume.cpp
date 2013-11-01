
#include "volume.h"

QVolume::QVolume(QObject* Parent /*= 0*/) :
	QSocketData(Parent)
{
}

QVolume::~QVolume()
{
}

void QVolume::Receive(QByteArray& Data)
{
}

void QVolume::Send(QByteArray& Data)
{
}