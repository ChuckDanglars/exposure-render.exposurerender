
#include "camerawidget.h"

QCameraWidget::QCameraWidget(QWidget* Parent) :
	QEditWidget(Parent),
	Camera(),
	Export(0),
	Import(0)
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

	this->Export = new QPushButton("Export");
	this->Import = new QPushButton("Import");

	this->layout()->addWidget(this->Export);
	this->layout()->addWidget(this->Import);

	connect(this->Export, SIGNAL(clicked()), this, SLOT(OnExport()));
	connect(this->Import, SIGNAL(clicked()), this, SLOT(OnImport()));
}

QCameraWidget::~QCameraWidget()
{
}

void QCameraWidget::OnExport()
{
	QString FileName = QFileDialog::getSaveFileName(this, "Export camera", QDir::currentPath(), "Exposure Render camera files (*.cam)");
	
	QFile CameraFile(FileName);
	CameraFile.open(QIODevice::WriteOnly);
	QDataStream Out(&CameraFile);

	Out << this->Camera;
	
	CameraFile.close();
}

void QCameraWidget::OnImport()
{
	QString FileName = QFileDialog::getOpenFileName(this, "Import camera", QDir::currentPath(), "Exposure Render camera files (*.cam)");
	
	QFile CameraFile(FileName);
	CameraFile.open(QIODevice::ReadOnly);
	QDataStream In(&CameraFile);

	In >> this->Camera;
	
	CameraFile.close();
}
