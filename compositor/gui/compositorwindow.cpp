
#include "compositorwindow.h"
#include "renderoutputwidget.h"
#include "server.h"

#include <QtGui>

QCompositorWindow::QCompositorWindow(QServer* Server, QWidget* Parent /*= 0*/, Qt::WindowFlags WindowFlags /*= 0*/) :
	QMainWindow(Parent, WindowFlags),
	Settings("compositor.ini", QSettings::IniFormat),
	CentralWidget(0),
	Server(Server),
	MainLayout(0),
	Connections(0),
	InteractionWidget(0),
	ConnectionItems(),
	RenderOutputWidget()
{
	setWindowTitle(tr("Compositor"));

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
	this->RenderOutputWidget->SetImage(this->Server->Estimate);
}

void QCompositorWindow::CreateStatusBar()
{
	this->AvgCombine = new QLabel("Combining estimates: ");
	this->AvgCombine->setFrameStyle(QFrame::Sunken);
    this->AvgCombine->setAlignment(Qt::AlignHCenter);
    this->AvgCombine->setMinimumSize(this->AvgCombine->sizeHint());


    statusBar()->addPermanentWidget(this->AvgCombine);
}

void QCompositorWindow::newThreadedSocket(QClientSocket* ClientSocket)
{
	QConnectionItem* ConnectionItem = new QConnectionItem(ClientSocket, this->Connections);

	this->ConnectionItems.append(ConnectionItem);

	connect(ConnectionItem, SIGNAL(remove(QConnectionItem*)), this, SLOT(removeConnectionItem(QConnectionItem*)));
}

void QCompositorWindow::removeConnectionItem(QConnectionItem* ConnectionItem)
{
	/*
	qDebug() << "Socket " << ThreadedSocket->Socket->peerAddress().toString() << " disconnected";
	
	
	QConnectionItem* ConnectionItem = this->ConnectionItems[ThreadedSocket];

	if (!ConnectionItem)
		return;

	this->Connections->takeItem(this->Connections->row(ConnectionItem));
	*/
}

void QCompositorWindow::OnUpdateEstimateIntegration(const float& EstimateIntegration)
{
	QString Text;
	Text.sprintf("Combining estimates: %0.2f ms", EstimateIntegration);
	this->AvgCombine->setText(Text);
}