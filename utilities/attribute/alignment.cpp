
#include "alignment.h"

QDataStream& operator << (QDataStream& Out, const QAlignmentAttribute& AlignmentAttribute)
{
	Out << AlignmentAttribute.Type;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QAlignmentAttribute& AlignmentAttribute)
{
	In >> AlignmentAttribute.Type;

    return In;
}

QAlignmentAttribute::QAlignmentAttribute(const QString& Name, const QString& Description, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Type("Type", "Type of alignment", 0, 0, QStringList() << "Look-at" << "Spherical")
{
}

QAlignmentAttribute::~QAlignmentAttribute()
{
}

void QAlignmentAttribute::Initialize()
{
	this->Type.Initialize();
}
