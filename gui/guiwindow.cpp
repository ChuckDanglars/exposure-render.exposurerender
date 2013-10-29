
#include "guiwindow.h"
#include "renderoutputwidget.h"
#include "clientsocket.h"

#include <QtGui>

QGuiWindow::QGuiWindow(QClientSocket* ClientSocket, QWidget* Parent /*= 0*/, Qt::WindowFlags WindowFlags /*= 0*/) :
	QMainWindow(Parent, WindowFlags),
	ClientSocket(ClientSocket),
	Settings("gui.ini", QSettings::IniFormat),
	CentralWidget(0),
	MainLayout(0),
	RenderOutputWidget()
{
	setWindowTitle(tr("Exposure Render GUI"));

	this->CentralWidget = new QWidget();

	this->setCentralWidget(this->CentralWidget);

	this->MainLayout = new QVBoxLayout();

	this->CentralWidget->setLayout(this->MainLayout);

	this->RenderOutputWidget = new QRenderOutputWidget();

	MainLayout->addWidget(this->RenderOutputWidget, 5);

	this->CreateStatusBar();

	QObject::connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnTimer()));

	this->Timer.start(1000.0f / this->Settings.value("gui/displayfps", 30).toInt());
}

QGuiWindow::~QGuiWindow()
{
}

void QGuiWindow::OnTimer()
{
//	this->RenderOutputWidget->SetImage(this->Server->Estimate);
}

void QGuiWindow::CreateStatusBar()
{
}