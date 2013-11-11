#ifndef QResource_H
#define QResource_H

#include "attribute\attributes.h"

class EXPOSURE_RENDER_DLL QResource : public QObject
{
    Q_OBJECT

public:
    QResource(QObject* Parent = 0);
    virtual ~QResource();

};

QDataStream& operator << (QDataStream& Out, const QResource& Resource);
QDataStream& operator >> (QDataStream& In, QResource& Resource);

#endif
