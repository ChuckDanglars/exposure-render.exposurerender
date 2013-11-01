
#include "float.h"

QFloatBinder::QFloatBinder(QAttribute* Attribute, QObject* Parent /*= 0*/) :
	QBinder(Attribute, Parent),
	ToMinimum(0),
	Decrement(0),
	Slide(0),
	Edit(0),
	Increment(0),
	ToMaximum(0),
	Reset(0)
{
	this->ToMinimum		= new QPushButton("[<");
	this->Decrement		= new QPushButton("<");
	this->Slide			= new QSlider(Qt::Horizontal);
	this->Edit			= new QDoubleSpinBox();
	this->Increment		= new QPushButton(">");
	this->ToMaximum		= new QPushButton(">]");
	this->Reset			= new QPushButton("r");

	this->Layout->addWidget(this->ToMinimum, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Decrement, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Slide, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Edit, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Increment, 0, Qt::AlignTop);
	this->Layout->addWidget(this->ToMaximum, 0, Qt::AlignTop);
	this->Layout->addWidget(this->Reset, 0, Qt::AlignTop);

	this->ToMinimum->setToolTip("Set the value to minimum");
	this->ToMinimum->setToolTip("Decrement the value one step");
	this->ToMinimum->setToolTip("Slide to change value");
	this->ToMinimum->setToolTip("Increment the value one step");
	this->ToMinimum->setToolTip("Edit the value manually");
	this->ToMinimum->setToolTip("Set the value to maximum");
	this->ToMinimum->setToolTip("Reset the value");

	this->ToMinimum->setFixedSize(20, 20);
	this->Decrement->setFixedSize(20, 20);
	this->Slide->setFixedHeight(20);
	this->Edit->setFixedHeight(20);
	this->Increment->setFixedSize(20, 20);
	this->ToMaximum->setFixedSize(20, 20);
	this->Reset->setFixedSize(20, 20);
}

QFloatBinder::~QFloatBinder()
{
}