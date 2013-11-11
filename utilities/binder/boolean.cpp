
#include "boolean.h"
#include "attribute\boolean.h"

QBooleanBinder::QBooleanBinder(QAttribute* Attribute, QWidget* Parent /*= 0*/) :
	QBinder(Attribute, Parent),
	CheckBox(0)
{
	this->CheckBox = new QCheckBox(this->GetAttribute()->GetName());

	this->Layout->addWidget(this->CheckBox, 0, Qt::AlignTop);

	connect(this->Attribute, SIGNAL(ValueChanged(bool)), this, SLOT(OnValueChanged(bool)));
	connect(this->CheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnCheckBoxStateChanged(int)));

	this->GetAttribute()->Initialize();
}

QBooleanBinder::~QBooleanBinder()
{
}

QBooleanAttribute* QBooleanBinder::GetAttribute()
{
	return (QBooleanAttribute*)this->Attribute;
}

void QBooleanBinder::OnValueChanged(bool Value)
{
	this->CheckBox->blockSignals(true);
	this->CheckBox->setChecked(Value);
	this->CheckBox->blockSignals(false);
}

void QBooleanBinder::OnCheckBoxStateChanged(int State)
{
	this->GetAttribute()->SetValue(State);
}
