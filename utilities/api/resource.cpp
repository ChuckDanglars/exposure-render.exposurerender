
#include "resource.h"

QDataStream& operator << (QDataStream& Out, const QResource& Resource)
{
    return Out;
}

QDataStream& operator >> (QDataStream& In, QResource& Resource)
{
    return In;
}

QResource::QResource(QObject* Parent /*= 0*/) :
	QObject(Parent)
{
}

QResource::~QResource()
{
}