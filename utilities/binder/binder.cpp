
#include "binder.h"

#include <QWidget>

QBinder::QBinder(QAttribute* Attribute, QObject* Parent /*= 0*/) :
	QObject(Parent),
	Attribute(Attribute),
	Widget(0),
	Layout(0),
	Label(0)
{
	this->Widget = new QWidget();

	this->Layout = new QHBoxLayout();

	this->Layout->setContentsMargins(0, 0, 0, 0);

	this->Widget->setLayout(this->Layout);

	this->Label = new QLabel(this->Attribute->GetName());

	this->Label->setFixedWidth(150);
	this->Label->setToolTip(this->Attribute->GetDescription());

	this->Layout->addWidget(this->Label);
}

QBinder::~QBinder()
{
}