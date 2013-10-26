
#include "rendererwindow.h"
#include "core\renderthread.h"
#include "core\renderer.h"

#include <QtGui>

QRendererWindow::QRendererWindow(QRenderer* Renderer, QWidget* Parent /*= 0*/, Qt::WindowFlags WindowFlags /*= 0*/) :
	QMainWindow(Parent, WindowFlags),
	Settings("renderer.ini", QSettings::IniFormat),
	Renderer(Renderer),
	RenderOutputWidget()
{
	setWindowTitle(tr("Renderer"));

	this->RenderOutputWidget = new QRenderOutputWidget();

	this->setCentralWidget(this->RenderOutputWidget);

	QObject::connect(&this->Timer, SIGNAL(timeout()), this, SLOT(OnTimer()));

	this->Timer.start(1000.0f / this->Settings.value("gui/displayfps", 30).toInt());
}

QRendererWindow::~QRendererWindow()
{
}

void QRendererWindow::OnTimer()
{
	this->RenderOutputWidget->SetImage(this->Renderer->Renderer.Camera.GetFilm().GetHostRunningEstimate());
}

void QRendererWindow::CreateStatusBar()
{
}