#ifndef QBind_H
#define QBind_H

#include "attribute\attribute.h"

using namespace Attributes;

namespace Binders
{

class EXPOSURE_RENDER_DLL QBinder : public QObject
{
    Q_OBJECT

public:
    QBinder(QAttribute* Attribute, QObject* Parent = 0);
    virtual ~QBinder();

private:
	QAttribute*		Attribute;
};

}

#endif
