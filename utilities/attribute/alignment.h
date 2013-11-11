#ifndef QAlignmentAttribute_H
#define QAlignmentAttribute_H

#include "attribute\option.h"

class EXPOSURE_RENDER_DLL QAlignmentAttribute : public QAttribute
{
    Q_OBJECT

public:
    QAlignmentAttribute(QObject* Parent = 0);
    virtual ~QAlignmentAttribute();

	/*
	Q_PROPERTY(QOptionAttribute Type READ GetType WRITE SetType NOTIFY ValueChanged)
		
	Q_PROPERTY(int DefaultValue READ GetDefaultValue WRITE SetDefaultValue)
	Q_PROPERTY(int Minimum READ GetMinimum WRITE SetMinimum NOTIFY MinimumChanged)
	Q_PROPERTY(int Maximum READ GetMaximum WRITE SetMaximum NOTIFY MaximumChanged)

	void SetValue(const int& Value)							{ this->Value = min(max(this->Minimum, Value), this->Maximum); emit ValueChanged(Value);		}
	int GetValue() const									{ return this->Value;																			}
	void ResetValue()										{ this->SetValue(this->DefaultValue); emit ValueChanged(Value);									}
	void SetDefaultValue(const int& DefaultValue)			{ this->DefaultValue = DefaultValue; 															}
	int GetDefaultValue() const								{ return this->DefaultValue;																	}
	void SetMinimum(const int& Minimum)						{ this->Minimum = min(Minimum, this->Maximum); emit MinimumChanged(Minimum);					}
	int GetMinimum() const									{ return this->Minimum;																			}
	void SetMaximum(const int& Maximum)						{ this->Maximum = max(Maximum, this->Minimum); emit MaximumChanged(Maximum);					}
	int GetMaximum() const									{ return this->Maximum;																			}
	void ToMinimum()										{ this->SetValue(this->GetMinimum());															}
	void ToMaximum()										{ this->SetValue(this->GetMaximum());															}
	void Decrement()										{ this->SetValue(this->GetValue() - 1);															}
	void Increment()										{ this->SetValue(this->GetValue() + 1);															}
	*/

	void Initialize();
	
signals:
	/*
	void ValueChanged(int);
    void MinimumChanged(int);
	void MaximumChanged(int);
	*/

protected:
	/*
	int		Value;
	int		DefaultValue;
	int		Minimum;
	int		Maximum;
	*/
	QOptionAttribute	Type;

	friend QDataStream& operator << (QDataStream& Out, const QAlignmentAttribute& AlignmentAttribute);
	friend QDataStream& operator >> (QDataStream& In, QAlignmentAttribute& AlignmentAttribute);
};

QDataStream& operator << (QDataStream& Out, const QAlignmentAttribute& AlignmentAttribute);
QDataStream& operator >> (QDataStream& In, QAlignmentAttribute& AlignmentAttribute);

#endif
