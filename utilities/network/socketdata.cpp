
#include "socketdata.h"

#include <QDebug>

QSocketData::QSocketData(QObject* Parent /*= 0*/) :
	QObject(Parent)
{
}

QSocketData::~QSocketData()
{
}

void QSocketData::Receive(QByteArray& Data)
{
	qDebug() << "Not implemented";
}

void QSocketData::Send(QByteArray& Data)
{
	qDebug() << "Not implemented";
}