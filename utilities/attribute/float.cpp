
#include "float.h"

namespace Attributes
{

QFloat::QFloat(const QString& Name, const QString& Description, const float& Value, const float& DefaultValue, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Value(Value),
	DefaultValue(DefaultValue)
{
}

QFloat::~QFloat()
{
}

}