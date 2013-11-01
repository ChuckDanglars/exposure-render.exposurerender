
#include "camerawidget.h"

QCameraWidget::QCameraWidget(QWidget* Parent) :
	QEditWidget(Parent),
	ApertureSize("Aperture size", "Lens opening size"),
	FieldOfView("FOV", "Field of view"),
	FocalDistance("Focal distance", "Focal distance")
{
	this->AddAttribute(&this->ApertureSize);
	this->AddAttribute(&this->FieldOfView);
	this->AddAttribute(&this->FocalDistance);

	this->Build();
}

QCameraWidget::~QCameraWidget()
{
}