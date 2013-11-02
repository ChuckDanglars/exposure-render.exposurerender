
#include "attribute.h"
#include "binder\binder.h"

QAttribute::QAttribute(const QString& Name, const QString& Description, QObject* Parent /*= 0*/) :
	QObject(Parent),
	Name(Name),
	Description(Description)
{
}

QAttribute::~QAttribute()
{
}