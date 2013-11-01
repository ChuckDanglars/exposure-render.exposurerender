
#include "attribute.h"

namespace Attributes
{

QAttribute::QAttribute(const QString& Name, const QString& Description, QObject* Parent /*= 0*/) :
	QObject(Parent),
	Name(Name),
	Description(Description)
{
}

QAttribute::~QAttribute()
{
}

}