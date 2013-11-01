
#include "guiwindow.h"
#include "utilities\gui\renderoutputwidget.h"
#include "network\compositorsocket.h"

#include <QtGui>

QGuiWindow::QGuiWindow(QCompositorSocket* CompositorSocket, QWidget* Parent /*= 0*/, Qt::WindowFlags WindowFlags /*= 0*/) :
	QMainWindow(Parent, WindowFlags),
	CompositorSocket(CompositorSocket),
	Settings("gui.ini", QSettings::IniFormat),
	CentralWidget(0),
	MainLayout(0),
	RenderOutputWidget(),
	UploadVolume(0),
	UploadBitmap(0)
{
	setWindowTitle(tr("Exposure Render GUI"));

	this->CentralWidget = new QWidget();

	this->setCentralWidget(this->CentralWidget);

	this->MainLayout = new QVBoxLayout();

	this->CentralWidget->setLayout(this->MainLayout);

	this->RenderOutputWidget = new QRenderOutputWidget();

	MainLayout->addWidget(this->RenderOutputWidget, 5);

	QObject::connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnTimer()));

	this->Timer.start(1000.0f / this->Settings.value("gui/displayfps", 30).toInt());

	this->UploadVolume	= new QPushButton("Upload volume");
	this->UploadBitmap	= new QPushButton("Upload bitmap");

	MainLayout->addWidget(this->UploadVolume);
	MainLayout->addWidget(this->UploadBitmap);

	connect(this->UploadVolume, SIGNAL(clicked()), this, SLOT(OnUploadVolume()));
	connect(this->UploadBitmap, SIGNAL(clicked()), this, SLOT(OnUploadBitmap()));

	connect(this->RenderOutputWidget, SIGNAL(CameraUpdate(float*, float*, float*)), this, SLOT(OnCameraUpdate(float*, float*, float*)));
}

QGuiWindow::~QGuiWindow()
{
}

void QGuiWindow::OnTimer()
{
	this->RenderOutputWidget->SetImage(this->CompositorSocket->Estimate.GetBuffer());
}

void QGuiWindow::OnUploadVolume()
{
	/*
	QString FileName = "C://workspaces//manix.raw";

	QFile File(FileName);

	if (File.open(QIODevice::ReadOnly))
	{
		char* Temp = (char*)malloc(File.size());

		memset(Temp, 0, File.size());

		File.read(Temp, File.size());

		QByteArray Voxels(Temp, File.size());
	
		QByteArray ByteArray;
		QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
		DataStream.setVersion(QDataStream::Qt_4_0);

		QFileInfo FileInfo(File);

		DataStream << FileInfo.fileName();
		
		DataStream << 256;
		DataStream << 230;
		DataStream << 256;

		DataStream << 1.0f;
		DataStream << 1.0f;
		DataStream << 1.0f;
		
		DataStream << Voxels;
	
		this->CompositorSocket->SendData("VOLUME", ByteArray);
	}
	else
	{
		qDebug() << "Unable to send volume!";
	}
	*/
}

void QGuiWindow::OnUploadBitmap()
{
	QString FileName = "C://workspaces//download.jpg";

	QFile File(FileName);

	if (File.open(QIODevice::ReadOnly))
	{
		QByteArray Voxels = File.readAll();
	
		QByteArray ByteArray;
		QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
		DataStream.setVersion(QDataStream::Qt_4_0);

		QFileInfo FileInfo(File);

		DataStream << FileInfo.fileName();
		DataStream << Voxels;
	
		this->CompositorSocket->SendData("BITMAP", ByteArray);
	}
	else
	{
		qDebug() << "Unable to send bitmap!";
	}
}

void QGuiWindow::OnCameraUpdate(float* Position, float* FocalPoint, float* ViewUp)
{
	QByteArray Data;
	QDataStream DataStream(&Data, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << Position[0];
	DataStream << Position[1];
	DataStream << Position[2];

	DataStream << FocalPoint[0];
	DataStream << FocalPoint[1];
	DataStream << FocalPoint[2];

	DataStream << ViewUp[0];
	DataStream << ViewUp[1];
	DataStream << ViewUp[2];

	this->CompositorSocket->SendData("CAMERA", Data);
}