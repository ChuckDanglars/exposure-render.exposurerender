
#include "emitter.h"

QDataStream& operator << (QDataStream& Out, const QEmitterAttribute& EmitterAttribute)
{
	Out << EmitterAttribute.Type;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QEmitterAttribute& EmitterAttribute)
{
	In >> EmitterAttribute.Type;

    return In;
}

QEmitterAttribute::QEmitterAttribute(QObject* Parent /*= 0*/) :
	QAttribute("Emitter", "Emitter", Parent),
	Type("Type", "Type of emitter", 0, 0, QStringList() << "" << "")
{
}

QEmitterAttribute::~QEmitterAttribute()
{
}

void QEmitterAttribute::Initialize()
{
	this->Type.Initialize();
}
