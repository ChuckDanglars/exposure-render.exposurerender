#ifndef QFloatAttribute_H
#define QFloatAttribute_H

#include "attribute\attribute.h"

namespace Attributes
{

class EXPOSURE_RENDER_DLL QFloat : public QAttribute
{
    Q_OBJECT

public:
    QFloat(const QString& Name, const QString& Description, const float& Value, const float& DefaultValue, QObject* Parent /*= 0*/);
    virtual ~QFloat();

private:
	float	Value;
	float	DefaultValue;
};

}

#endif
