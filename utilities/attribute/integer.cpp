
#include "integer.h"

QDataStream& operator << (QDataStream& Out, const QIntegerAttribute& IntegerAttribute)
{
	Out << IntegerAttribute.Minimum;
	Out << IntegerAttribute.Maximum;
	Out << IntegerAttribute.Value;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QIntegerAttribute& IntegerAttribute)
{
	In >> IntegerAttribute.Minimum;
	In >> IntegerAttribute.Maximum;
	In >> IntegerAttribute.Value;

    return In;
}

QIntegerAttribute::QIntegerAttribute(const QString& Name, const QString& Description, const int& Value /*= 0.0f*/, const int& DefaultValue /*= 0.0f*/, const int& Minimum /*= 0*/, const int& Maximum /*= 100*/, QObject* Parent /*= 0*/) :
	QAttribute(Name, Description, Parent),
	Value(Value),
	DefaultValue(DefaultValue),
	Minimum(Minimum),
	Maximum(Maximum)
{
}

QIntegerAttribute::~QIntegerAttribute()
{
}

void QIntegerAttribute::Initialize()
{
	emit this->MinimumChanged(this->GetMinimum());
	emit this->MaximumChanged(this->GetMaximum());
	emit this->ValueChanged(this->GetValue());
}
