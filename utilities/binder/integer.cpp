
#include "integer.h"
#include "attribute\integer.h"

QIntegerBinder::QIntegerBinder(QAttribute* Attribute, const bool& Tracking /*= false*/, QWidget* Parent /*= 0*/) :
	QBinder(Attribute, Parent),
	ToMinimum(0),
	Decrement(0),
	Slide(0),
	Increment(0),
	ToMaximum(0),
	Edit(0),
	Reset(0)
{
	this->ToMinimum		= new QPushButton("[");
	this->Decrement		= new QPushButton("<");
	this->Slide			= new QSlider(Qt::Horizontal);
	this->Increment		= new QPushButton(">");
	this->ToMaximum		= new QPushButton("]");
	this->Edit			= new QSpinBox();
	this->Reset			= new QPushButton("r");

	this->Layout->addWidget(this->ToMinimum, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Decrement, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Slide, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Increment, 0, Qt::AlignTop);
	this->Layout->addWidget(this->ToMaximum, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Edit, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Reset, 0, Qt::AlignTop);

	this->Slide->setTracking(Tracking);

	this->ToMinimum->setToolTip("Set the value to minimum");
	this->Decrement->setToolTip("Decrement the value one step");
	this->Slide->setToolTip("Slide to change value");
	this->Increment->setToolTip("Increment the value one step");
	this->ToMaximum->setToolTip("Edit the value manually");
	this->Edit->setToolTip("Set the value to maximum");
	this->Reset->setToolTip("Reset the value");

	this->ToMinimum->setFixedSize(20, 20);
	this->Decrement->setFixedSize(20, 20);
	this->Slide->setFixedHeight(20);
	this->Edit->setFixedHeight(20);
	this->Edit->setFixedWidth(75);
	this->Increment->setFixedSize(20, 20);
	this->ToMaximum->setFixedSize(20, 20);
	this->Reset->setFixedSize(20, 20);

	connect(this->Attribute, SIGNAL(ValueChanged(int)), this, SLOT(OnValueChanged(int)));
	connect(this->Attribute, SIGNAL(MinimumChanged(int)), this, SLOT(OnMinimumChanged(int)));
	connect(this->Attribute, SIGNAL(MaximumChanged(int)), this, SLOT(OnMaximumChanged(int)));
	connect(this->Slide, SIGNAL(valueChanged(int)), this, SLOT(OnSliderValueChanged(int)));
	connect(this->Edit, SIGNAL(valueChanged(int)), this, SLOT(OnEditValueChanged(int)));
	connect(this->Reset, SIGNAL(clicked()), this, SLOT(OnResetValue()));
	connect(this->ToMinimum, SIGNAL(clicked()), this, SLOT(OnToMinimum()));
	connect(this->Decrement, SIGNAL(clicked()), this, SLOT(OnDecrement()));
	connect(this->Increment, SIGNAL(clicked()), this, SLOT(OnIncrement()));
	connect(this->ToMaximum, SIGNAL(clicked()), this, SLOT(OnToMaximum()));

	this->GetAttribute()->Initialize();
}

QIntegerBinder::~QIntegerBinder()
{
}

QIntegerAttribute* QIntegerBinder::GetAttribute()
{
	return (QIntegerAttribute*)this->Attribute;
}

void QIntegerBinder::OnValueChanged(int Value)
{
	this->Slide->blockSignals(true);
	this->Slide->setValue(Value);
	this->Slide->blockSignals(false);

	this->Edit->blockSignals(true);
	this->Edit->setValue(Value);
	this->Edit->blockSignals(false);
}

void QIntegerBinder::OnMinimumChanged(int Value)
{
	this->Slide->setMinimum(this->GetAttribute()->GetMinimum());
	this->Edit->setMinimum(this->GetAttribute()->GetMinimum());
}

void QIntegerBinder::OnMaximumChanged(int Value)
{
	this->Slide->setMaximum(this->GetAttribute()->GetMaximum());
	this->Edit->setMaximum(this->GetAttribute()->GetMaximum());
}

void QIntegerBinder::OnSliderValueChanged(int Value)
{
	this->GetAttribute()->SetValue(Value);
}

void QIntegerBinder::OnEditValueChanged(int Value)
{
	this->GetAttribute()->SetValue(Value);
}

void QIntegerBinder::OnResetValue()
{
	this->GetAttribute()->ResetValue();
}

void QIntegerBinder::OnToMinimum()
{
	this->GetAttribute()->ToMinimum();
}

void QIntegerBinder::OnDecrement()
{
	this->GetAttribute()->Decrement();
}

void QIntegerBinder::OnIncrement()
{
	this->GetAttribute()->Increment();
}

void QIntegerBinder::OnToMaximum()
{
	this->GetAttribute()->ToMaximum();
}