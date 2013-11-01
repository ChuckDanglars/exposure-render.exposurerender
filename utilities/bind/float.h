#ifndef QFloatBinder_H
#define QFloatBinder_H

#include "attribute\attribute.h"

using namespace Attributes;

namespace Binders
{

class EXPOSURE_RENDER_DLL QFloatBinder : public QObject
{
    Q_OBJECT

public:
    QFloatBinder(QAttribute* Attribute, QObject* Parent = 0);
    virtual ~QFloatBinder();

private:
	QAttribute*		Attribute;
};

}

#endif
