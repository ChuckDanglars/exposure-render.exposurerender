
#include "binder.h"

#include <QWidget>

QBinder::QBinder(QAttribute* Attribute, QWidget* Parent /*= 0*/) :
	QWidget(Parent),
	Attribute(Attribute),
	Layout(0),
	Label(0)
{
	this->Layout = new QHBoxLayout();

	this->Layout->setContentsMargins(0, 0, 0, 0);

	this->setLayout(this->Layout);

	this->Label = new QLabel(this->Attribute->GetName());

	this->Label->setFixedWidth(150);
	this->Label->setToolTip(this->Attribute->GetDescription());

	this->Layout->addWidget(this->Label);
}

QBinder::~QBinder()
{
}