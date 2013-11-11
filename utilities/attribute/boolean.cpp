
#include "boolean.h"

QDataStream& operator << (QDataStream& Out, const QBooleanAttribute& BooleanAttribute)
{
	Out << BooleanAttribute.Value;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QBooleanAttribute& BooleanAttribute)
{
	In >> BooleanAttribute.Value;

    return In;
}

QBooleanAttribute::QBooleanAttribute(const QString& Name, const QString& Description, const bool& Value /*= false*/, const bool& DefaultValue /*= false*/, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Value(Value),
	DefaultValue(DefaultValue)
{
}

QBooleanAttribute::~QBooleanAttribute()
{
}

void QBooleanAttribute::Initialize()
{
	emit this->ValueChanged(this->GetValue());
}