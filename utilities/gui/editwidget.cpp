
#include "editwidget.h"

QEditWidget::QEditWidget(QWidget* Parent) :
	QWidget(Parent)
{
	this->Layout = new QVBoxLayout();

	this->setLayout(this->Layout);

	this->Layout->setContentsMargins(0, 0, 0, 0);
}

QEditWidget::~QEditWidget()
{
}