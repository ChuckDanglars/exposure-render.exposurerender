
#include "option.h"

QDataStream& operator << (QDataStream& Out, const QOptionAttribute& OptionAttribute)
{
	Out << OptionAttribute.Strings;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QOptionAttribute& OptionAttribute)
{
	In >> OptionAttribute.Strings;

    return In;
}

QOptionAttribute::QOptionAttribute(const QString& Name, const QString& Description, const int& Value /*= 0.0f*/, const int& DefaultValue /*= 0.0f*/, const QStringList& Strings /*= QStringList()*/, QObject* Parent /*= 0*/) :
	QIntegerAttribute(Name, Description, Value, DefaultValue, 0, 100, Parent),
	Strings(Strings)
{
}

QOptionAttribute::~QOptionAttribute()
{
}

void QOptionAttribute::Initialize()
{
	QIntegerAttribute::Initialize();

	emit this->StringsChanged(this->GetStrings());
}
