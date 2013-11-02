
#include "float.h"
#include "binder\float.h"

QFloatAttribute::QFloatAttribute(const QString& Name, const QString& Description, const float& Value /*= 0.0f*/, const float& DefaultValue /*= 0.0f*/, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Value(Value),
	DefaultValue(DefaultValue)
{
}

QFloatAttribute::~QFloatAttribute()
{
}