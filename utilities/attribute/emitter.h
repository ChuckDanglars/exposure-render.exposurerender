#ifndef QEmitterAttribute_H
#define QEmitterAttribute_H

#include "attribute\option.h"

class EXPOSURE_RENDER_DLL QEmitterAttribute : public QAttribute
{
    Q_OBJECT

public:
    QEmitterAttribute(QObject* Parent = 0);
    virtual ~QEmitterAttribute();

	void Initialize();
	
signals:

protected:
	QOptionAttribute	Type;

	friend QDataStream& operator << (QDataStream& Out, const QEmitterAttribute& EmitterAttribute);
	friend QDataStream& operator >> (QDataStream& In, QEmitterAttribute& EmitterAttribute);
};

QDataStream& operator << (QDataStream& Out, const QEmitterAttribute& EmitterAttribute);
QDataStream& operator >> (QDataStream& In, QEmitterAttribute& EmitterAttribute);

#endif
