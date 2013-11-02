
#include "float.h"

QFloatAttribute::QFloatAttribute(const QString& Name, const QString& Description, const float& Value /*= 0.0f*/, const float& DefaultValue /*= 0.0f*/, const float& Minimum /*= 0.0f*/, const float& Maximum /*= 100.0f*/, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Value(Value),
	DefaultValue(DefaultValue),
	Minimum(Minimum),
	Maximum(Maximum)
{
}

QFloatAttribute::~QFloatAttribute()
{
}

void QFloatAttribute::Initialize()
{
	emit this->MinimumChanged(this->GetMinimum());
	emit this->MaximumChanged(this->GetMaximum());
	emit this->ValueChanged(this->GetValue());
}