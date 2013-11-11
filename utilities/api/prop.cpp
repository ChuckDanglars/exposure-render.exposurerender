
#include "prop.h"

QDataStream& operator << (QDataStream& Out, const QProp& Prop)
{
    return Out;
}

QDataStream& operator >> (QDataStream& In, QProp& Prop)
{
    return In;
}

QProp::QProp(QObject* Parent /*= 0*/) :
	QObject(Parent)
{
}

QProp::~QProp()
{
}