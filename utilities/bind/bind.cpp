
#include "binder.h"

using namespace Attributes;

namespace Binders
{

QBinder::QBinder(QAttribute* Attribute, QObject* Parent /*= 0*/) :
	QObject(Parent),
	Attribute(Attribute)
{
}

QBinder::~QBinder()
{
}

}