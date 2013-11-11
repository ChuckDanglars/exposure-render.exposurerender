
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
	connect(this->ComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxIndexValueChanged(int)));
	
	connect(this->First, SIGNAL(clicked()), this, SLOT(OnFirst()));
	connect(this->Previous, SIGNAL(clicked()), this, SLOT(OnPrevious()));
	connect(this->Next, SIGNAL(clicked()), this, SLOT(OnNext()));
	connect(this->Last, SIGNAL(clicked()), this, SLOT(OnLast()));
	connect(this->Reset, SIGNAL(clicked()), this, SLOT(OnResetValue()));

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
	this->ComboBox->blockSignals(true);
	this->ComboBox->setCurrentIndex(Value);
	this->ComboBox->blockSignals(false);
}

void QOptionBinder::OnComboBoxIndexValueChanged(int Value)
{
	this->GetAttribute()->SetValue(Value);
}

void QOptionBinder::OnFirst()
{
	this->GetAttribute()->ToMinimum();
}

void QOptionBinder::OnPrevious()
{
	this->GetAttribute()->Decrement();
}

void QOptionBinder::OnNext()
{
	this->GetAttribute()->Increment();
}

void QOptionBinder::OnLast()
{
	this->GetAttribute()->ToMaximum();
}

void QOptionBinder::OnResetValue()
{
	this->GetAttribute()->ResetValue();
}