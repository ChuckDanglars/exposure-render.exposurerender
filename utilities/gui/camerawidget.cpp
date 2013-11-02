
#include "camerawidget.h"

QCameraWidget::QCameraWidget(QWidget* Parent) :
	QEditWidget(Parent),
	ApertureSize("Aperture size", "Lens opening size"),
	FieldOfView("FOV", "Field of view"),
	FocalDistance("Focal distance", "Focal distance"),
	FilmWidth("Width", "Film width", 1024, 1024, 0, 2048),
	FilmHeight("Height", "Film height", 768, 768, 0, 2048),
	Exposure("Exposure", "Film exposure"),
	Gamma("Gamma", "Film gamma")
{
	QGroupBox* General = new QGroupBox("General");

	General->setLayout(new QVBoxLayout());

	General->layout()->addWidget(new QFloatBinder(&this->ApertureSize, this));
	General->layout()->addWidget(new QFloatBinder(&this->FieldOfView, this));
	General->layout()->addWidget(new QFloatBinder(&this->FocalDistance, this));

	this->layout()->addWidget(General);

	QGroupBox* Film = new QGroupBox("Film");

	Film->setLayout(new QVBoxLayout());

	Film->layout()->addWidget(new QIntegerBinder(&this->FilmWidth, this));
	Film->layout()->addWidget(new QIntegerBinder(&this->FilmHeight, this));
	Film->layout()->addWidget(new QFloatBinder(&this->Exposure, this));
	Film->layout()->addWidget(new QFloatBinder(&this->Gamma, this));

	this->layout()->addWidget(Film);
}

QCameraWidget::~QCameraWidget()
{
}