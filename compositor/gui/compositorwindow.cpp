
#include "compositorwindow.h"
#include "utilities\gui\renderoutputwidget.h"
#include "server\rendererserver.h"
#include "server\guiserver.h"

#include <QtGui>

QCompositorWindow::QCompositorWindow(QRendererServer* RendererServer, QGuiServer* GuiServer, QWidget* Parent /*= 0*/, Qt::WindowFlags WindowFlags /*= 0*/) :
	QMainWindow(Parent, WindowFlags),
	Settings("compositor.ini", QSettings::IniFormat),
	RendererServer(RendererServer),
	GuiServer(GuiServer),
	CentralWidget(0),
	MainLayout(0),
	Connections(0),
	InteractionWidget(0),
	RenderOutputWidget()
{
	setWindowTitle(tr("Exposure Render compositor"));

	this->CentralWidget = new QWidget();

	this->setCentralWidget(this->CentralWidget);

	this->MainLayout = new QVBoxLayout();

	this->CentralWidget->setLayout(this->MainLayout);

	this->Connections = new QTreeWidget();

	this->Connections->setRootIsDecorated(false);
	this->Connections->setColumnCount(7);
	this->Connections->setColumnWidth(0, 70);
	this->Connections->setColumnWidth(1, 85);
	this->Connections->setColumnWidth(2, 85);
	this->Connections->setColumnWidth(3, 85);
	this->Connections->setColumnWidth(4, 85);
	this->Connections->setColumnWidth(5, 85);
	this->Connections->setHeaderLabels(QStringList() << "IP Address" << "Status" << "Image size" << "Performance" << "JPG Encoding" << "JPG Decoding" << "Device");
	this->Connections->header()->setStretchLastSection(true);

	MainLayout->addWidget(this->Connections, 1);

	this->RenderOutputWidget = new QRenderOutputWidget();

	MainLayout->addWidget(this->RenderOutputWidget, 5);

	this->CreateStatusBar();

	QObject::connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnTimer()));

	this->Timer.start(1000.0f / this->Settings.value("gui/displayfps", 30).toInt());
}

QCompositorWindow::~QCompositorWindow()
{
}

void QCompositorWindow::OnTimer()
{
//	this->RenderOutputWidget->SetImage(this->RendererServer->Estimate);
}

void QCompositorWindow::CreateStatusBar()
{
}