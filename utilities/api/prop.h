#ifndef QProp_H
#define QProp_H

#include "attribute\attributes.h"

class EXPOSURE_RENDER_DLL QProp : public QObject
{
    Q_OBJECT

public:
    QProp(QObject* Parent = 0);
    virtual ~QProp();

};

QDataStream& operator << (QDataStream& Out, const QProp& Prop);
QDataStream& operator >> (QDataStream& In, QProp& Prop);

#endif
