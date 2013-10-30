
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
}

QGuiWindow::~QGuiWindow()
{
}

void QGuiWindow::OnTimer()
{
//	this->RenderOutputWidget->SetImage(this->Server->Estimate);
}

void QGuiWindow::OnUploadVolume()
{
	QString FileName = "C://workspaces//manix.raw";

	QFile File(FileName);
	File.open(QIODevice::ReadOnly);
	
	QByteArray Voxels = File.readAll();

	/*
	QByteArray ByteArray;
	QDataStream DataStream(&ByteArray, QIODevice::ReadWrite);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << QString("asdsadsdsadad");
	//DataStream << Voxels;
	*/

	// qDebug() << "Sending" << Action << " of " << Data.count() << "bytes";

	QByteArray ByteArray;

	QDataStream DataStream(&ByteArray, QIODevice::WriteOnly);
	DataStream.setVersion(QDataStream::Qt_4_0);

	DataStream << quint32(0);

	DataStream << QString("VOLUME");
	DataStream << FileName;

	DataStream.device()->seek(0);
		    
	DataStream << (quint32)(ByteArray.size() - sizeof(quint32));

    this->CompositorSocket->write(ByteArray);
	this->CompositorSocket->flush();




	
	// this->CompositorSocket->SendData("VOLUME", ByteArray);
}

void QGuiWindow::OnUploadBitmap()
{
}