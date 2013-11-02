
#include "camerawidget.h"

QCameraWidget::QCameraWidget(QWidget* Parent) :
	QEditWidget(Parent),
	Camera()
{
	QGroupBox* General = new QGroupBox("General");

	General->setLayout(new QVBoxLayout());

	General->layout()->addWidget(new QFloatBinder(&this->Camera.ApertureSize, this));
	General->layout()->addWidget(new QFloatBinder(&this->Camera.FieldOfView, this));
	General->layout()->addWidget(new QFloatBinder(&this->Camera.FocalDistance, this));

	this->layout()->addWidget(General);

	QGroupBox* Film = new QGroupBox("Film");

	Film->setLayout(new QVBoxLayout());

	Film->layout()->addWidget(new QIntegerBinder(&this->Camera.FilmWidth, this));
	Film->layout()->addWidget(new QIntegerBinder(&this->Camera.FilmHeight, this));
	Film->layout()->addWidget(new QFloatBinder(&this->Camera.Exposure, this));
	Film->layout()->addWidget(new QFloatBinder(&this->Camera.Gamma, this));

	this->layout()->addWidget(Film);
}

QCameraWidget::~QCameraWidget()
{
}