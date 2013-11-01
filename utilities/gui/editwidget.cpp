
#include "editwidget.h"

QEditWidget::QEditWidget(QWidget* Parent) :
	QWidget(Parent),
	Attributes()
{
	this->Layout = new QVBoxLayout();

	this->setLayout(this->Layout);

	this->Layout->setContentsMargins(0, 0, 0, 0);
}

QEditWidget::~QEditWidget()
{
}

void QEditWidget::Build()
{
	foreach (QAttribute* Attribute, this->Attributes)
	{
		this->Layout->addWidget(Attribute->GetWidget());
	}
}

void QEditWidget::AddAttribute(QAttribute* Attribute)
{
	this->Attributes.append(Attribute);
}