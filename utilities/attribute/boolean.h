#ifndef QFloatAttribute_H
#define QFloatAttribute_H

#include "attribute\attribute.h"

class EXPOSURE_RENDER_DLL QBooleanAttribute : public QAttribute
{
    Q_OBJECT

public:
    QBooleanAttribute(const QString& Name, const QString& Description, const bool& Value = false, const bool& DefaultValue = false, QObject* Parent = 0);
    virtual ~QBooleanAttribute();

	Q_PROPERTY(float Value READ GetValue WRITE SetValue RESET ResetValue NOTIFY ValueChanged)
	Q_PROPERTY(float DefaultValue READ GetDefaultValue WRITE SetDefaultValue)

	void SetValue(const bool& Value)								{ this->Value = Value; emit ValueChanged(Value);												}
	bool GetValue() const											{ return this->Value;																			}
	void ResetValue()												{ this->SetValue(this->DefaultValue); emit ValueChanged(Value);									}
	void SetDefaultValue(const bool& DefaultValue)					{ this->DefaultValue = DefaultValue;															}
	bool GetDefaultValue() const									{ return this->DefaultValue;																	}

	void Initialize();

signals:
	void ValueChanged(bool);

private:
	bool		Value;
	bool		DefaultValue;
};

QDataStream& operator << (QDataStream& Out, const QBooleanAttribute& BooleanAttribute);
QDataStream& operator >> (QDataStream& In, QBooleanAttribute& BooleanAttribute);

#endif
