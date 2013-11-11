#ifndef QShapeAttribute_H
#define QShapeAttribute_H

#include "attribute\option.h"

class EXPOSURE_RENDER_DLL QShapeAttribute : public QAttribute
{
    Q_OBJECT

public:
    QShapeAttribute(QObject* Parent = 0);
    virtual ~QShapeAttribute();

	void Initialize();
	
signals:

protected:
	QOptionAttribute	Type;

	friend QDataStream& operator << (QDataStream& Out, const QShapeAttribute& ShapeAttribute);
	friend QDataStream& operator >> (QDataStream& In, QShapeAttribute& ShapeAttribute);
};

QDataStream& operator << (QDataStream& Out, const QShapeAttribute& ShapeAttribute);
QDataStream& operator >> (QDataStream& In, QShapeAttribute& ShapeAttribute);

#endif
