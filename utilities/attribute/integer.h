#ifndef QIntegerAttribute_H
#define QIntegerAttribute_H

#include "attribute\attribute.h"

class EXPOSURE_RENDER_DLL QIntegerAttribute : public QAttribute
{
    Q_OBJECT

public:
    QIntegerAttribute(const QString& Name, const QString& Description, const int& Value = 0, const int& DefaultValue = 0, const int& Minimum = 0, const int& Maximum = 100, QObject* Parent = 0);
    virtual ~QIntegerAttribute();

	Q_PROPERTY(int Value READ GetValue WRITE SetValue RESET ResetValue NOTIFY ValueChanged)
	Q_PROPERTY(int DefaultValue READ GetDefaultValue WRITE SetDefaultValue)
	Q_PROPERTY(int Minimum READ GetMinimum WRITE SetMinimum NOTIFY MinimumChanged)
	Q_PROPERTY(int Maximum READ GetMaximum WRITE Setmaximum NOTIFY MaximumChanged)

	void SetValue(const int& Value)							{ this->Value = min(max(this->Minimum, Value), this->Maximum); emit ValueChanged(Value);		}
	int GetValue() const									{ return this->Value;																			}
	void ResetValue()										{ this->SetValue(this->DefaultValue); emit ValueChanged(Value);									}
	void SetDefaultValue(const int& DefaultValue)			{ this->DefaultValue = DefaultValue; 															}
	int GetDefaultValue() const								{ return this->DefaultValue;																	}
	void SetMinimum(const int& Minimum)						{ this->Minimum = min(Minimum, this->Maximum); emit MinimumChanged(Minimum);					}
	int GetMinimum() const									{ return this->Minimum;																			}
	void Setmaximum(const int& maximum)						{ this->Maximum = max(Maximum, this->Minimum); emit MaximumChanged(Maximum);					}
	int GetMaximum() const									{ return this->Maximum;																			}
	void ToMinimum()										{ this->SetValue(this->GetMinimum());															}
	void ToMaximum()										{ this->SetValue(this->GetMaximum());															}
	void Decrement()										{ this->SetValue(this->GetValue() - 1);															}
	void Increment()										{ this->SetValue(this->GetValue() + 1);															}

	void Initialize();
	
signals:
	void ValueChanged(int);
    void MinimumChanged(int);
	void MaximumChanged(int);

private:
	int		Value;
	int		DefaultValue;
	int		Minimum;
	int		Maximum;
};

#endif
