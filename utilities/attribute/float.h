#ifndef QFloatAttribute_H
#define QFloatAttribute_H

#include "attribute\attribute.h"

class EXPOSURE_RENDER_DLL QFloatAttribute : public QAttribute
{
    Q_OBJECT

public:
    QFloatAttribute(const QString& Name, const QString& Description, const float& Value = 0.0f, const float& DefaultValue = 0.0f, QObject* Parent = 0);
    virtual ~QFloatAttribute();

private:
	float	Value;
	float	DefaultValue;
};

#endif
