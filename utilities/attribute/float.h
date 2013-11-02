#ifndef QFloatAttribute_H
#define QFloatAttribute_H

#include "attribute\attribute.h"

class EXPOSURE_RENDER_DLL QFloatAttribute : public QAttribute
{
    Q_OBJECT

public:
    QFloatAttribute(const QString& Name, const QString& Description, const float& Value = 0.0f, const float& DefaultValue = 0.0f, const float& Minimum = 0.0f, const float& Maximum = 100.0f, QObject* Parent = 0);
    virtual ~QFloatAttribute();

	Q_PROPERTY(float Value READ GetValue WRITE SetValue RESET ResetValue NOTIFY ValueChanged)
	Q_PROPERTY(float DefaultValue READ GetDefaultValue WRITE SetDefaultValue)
	Q_PROPERTY(float Minimum READ GetMinimum WRITE SetMinimum NOTIFY MinimumChanged)
	Q_PROPERTY(float Maximum READ GetMaximum WRITE SetMaximum NOTIFY MaximumChanged)

	void SetValue(const float& Value)							{ this->Value = min(max(this->Minimum, Value), this->Maximum); emit ValueChanged(Value);		}
	float GetValue() const										{ return this->Value;																			}
	void ResetValue()											{ this->SetValue(this->DefaultValue); emit ValueChanged(Value);									}
	void SetDefaultValue(const float& DefaultValue)				{ this->DefaultValue = DefaultValue;															}
	float GetDefaultValue() const								{ return this->DefaultValue;																	}
	void SetMinimum(const float& Minimum)						{ this->Minimum = min(Minimum, this->Maximum); emit MinimumChanged(Minimum);					}
	float GetMinimum() const									{ return this->Minimum;																			}
	void SetMaximum(const float& Maximum)						{ this->Maximum = max(Maximum, this->Minimum); emit MaximumChanged(Maximum);					}
	float GetMaximum() const									{ return this->Maximum;																			}
	void ToMinimum()											{ this->SetValue(this->GetMinimum());															}
	void ToMaximum()											{ this->SetValue(this->GetMaximum());															}
	void Decrement()											{ this->SetValue(this->GetValue() - 1);															}
	void Increment()											{ this->SetValue(this->GetValue() + 1);															}

	void Initialize();

signals:
	void ValueChanged(float);
    void MinimumChanged(float);
	void MaximumChanged(float);

private:
	float	Value;
	float	DefaultValue;
	float	Minimum;
	float	Maximum;
};

QDataStream& operator << (QDataStream& Out, const QFloatAttribute& FloatAttribute);
QDataStream& operator >> (QDataStream& In, QFloatAttribute& FloatAttribute);

#endif
