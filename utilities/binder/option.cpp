
#include "option.h"
#include "attribute\option.h"

QOptionBinder::QOptionBinder(QAttribute* Attribute, QWidget* Parent /*= 0*/) :
	QBinder(Attribute, Parent),
	ComboBox(0),
	First(0),
	Previous(0),
	Next(0),
	Last(0),
	Reset(0)
{
	this->ComboBox		= new QComboBox();
	this->First			= new QPushButton("[");
	this->Previous		= new QPushButton("<");
	this->Next			= new QPushButton(">");
	this->Last			= new QPushButton("]");
	this->Reset			= new QPushButton("r");

	this->Layout->addWidget(this->ComboBox, 0, Qt::AlignTop);
	this->Layout->addWidget(this->First, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Previous, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Next, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Last, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Reset, 0, Qt::AlignTop);

	this->ComboBox->setToolTip("Select option");
	this->First->setToolTip("Select the first option");
	this->Previous->setToolTip("Select the previous option");
	this->Next->setToolTip("Select the next option");
	this->Last->setToolTip("Select the last option");
	this->Reset->setToolTip("Reset the value");

	this->First->setFixedSize(20, 20);
	this->Previous->setFixedSize(20, 20);
	this->Next->setFixedHeight(20);
	this->Last->setFixedHeight(20);
	this->Reset->setFixedWidth(75);
	
	connect(this->Attribute, SIGNAL(ValueChanged(int)), this, SLOT(OnValueChanged(int)));
	/*
	connect(this->Slide, SIGNAL(valueChanged(int)), this, SLOT(OnSliderValueChanged(int)));
	connect(this->Edit, SIGNAL(valueChanged(int)), this, SLOT(OnEditValueChanged(int)));
	connect(this->Reset, SIGNAL(clicked()), this, SLOT(OnResetValue()));
	connect(this->ToMinimum, SIGNAL(clicked()), this, SLOT(OnToMinimum()));
	connect(this->Decrement, SIGNAL(clicked()), this, SLOT(OnDecrement()));
	connect(this->Increment, SIGNAL(clicked()), this, SLOT(OnIncrement()));
	connect(this->ToMaximum, SIGNAL(clicked()), this, SLOT(OnToMaximum()));
	*/

	this->GetAttribute()->Initialize();
}

QOptionBinder::~QOptionBinder()
{
}

QOptionAttribute* QOptionBinder::GetAttribute()
{
	return (QOptionAttribute*)this->Attribute;
}

void QOptionBinder::OnValueChanged(int Value)
{
	/*
	this->Slide->blockSignals(true);
	this->Slide->setValue(Value);
	this->Slide->blockSignals(false);

	this->Edit->blockSignals(true);
	this->Edit->setValue(Value);
	this->Edit->blockSignals(false);
	*/
}

/*
void QOptionBinder::OnMinimumChanged(int Value)
{
	this->Slide->setMinimum(this->GetAttribute()->GetMinimum());
	this->Edit->setMinimum(this->GetAttribute()->GetMinimum());
}

void QOptionBinder::OnMaximumChanged(int Value)
{
	this->Slide->setMaximum(this->GetAttribute()->GetMaximum());
	this->Edit->setMaximum(this->GetAttribute()->GetMaximum());
}

void QOptionBinder::OnSliderValueChanged(int Value)
{
	this->GetAttribute()->SetValue(Value);
}

void QOptionBinder::OnEditValueChanged(int Value)
{
	this->GetAttribute()->SetValue(Value);
}

void QOptionBinder::OnResetValue()
{
	this->GetAttribute()->ResetValue();
}

void QOptionBinder::OnToMinimum()
{
	this->GetAttribute()->ToMinimum();
}

void QOptionBinder::OnDecrement()
{
	this->GetAttribute()->Decrement();
}

void QOptionBinder::OnIncrement()
{
	this->GetAttribute()->Increment();
}

void QOptionBinder::OnToMaximum()
{
	this->GetAttribute()->ToMaximum();
}
*/