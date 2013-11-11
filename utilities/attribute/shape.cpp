
#include "shape.h"

QDataStream& operator << (QDataStream& Out, const QShapeAttribute& ShapeAttribute)
{
	Out << ShapeAttribute.Type;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QShapeAttribute& ShapeAttribute)
{
	In >> ShapeAttribute.Type;

    return In;
}

QShapeAttribute::QShapeAttribute(QObject* Parent /*= 0*/) :
	QAttribute("Shape", "Shape", Parent),
	Type("Type", "Type of emitter", 0, 0, QStringList() << "" << "")
{
}

QShapeAttribute::~QShapeAttribute()
{
}

void QShapeAttribute::Initialize()
{
	this->Type.Initialize();
}
