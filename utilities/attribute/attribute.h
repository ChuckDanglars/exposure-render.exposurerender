#ifndef QAttribute_H
#define QAttribute_H

#include "general\define.h"

#include <QtGui>

namespace Attributes
{

class EXPOSURE_RENDER_DLL QAttribute : public QObject
{
    Q_OBJECT

public:
    QAttribute(const QString& Name, const QString& Description, QObject* Parent = 0);
    virtual ~QAttribute();

private:
	QString		Name;
	QString		Description;
};

}

#endif
