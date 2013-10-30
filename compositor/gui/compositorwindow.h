#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QtNetwork>

class QRendererServer;
class QGuiServer;
class QInteractionWidget;
class QRenderOutputWidget;

class QCompositorWindow : public QMainWindow
{
    Q_OBJECT

public:
	QCompositorWindow(QRendererServer* RendererServer, QGuiServer* GuiServer, QWidget* Parent = 0, Qt::WindowFlags WindowFlags = 0);
	virtual ~QCompositorWindow();

	void CreateStatusBar();
	QRenderOutputWidget* GetRenderOutputWidget() { return this->RenderOutputWidget; }

public slots:
	void OnTimer();

private:
	QSettings					Settings;
	
	QRendererServer*			RendererServer;
	QGuiServer*					GuiServer;
	QWidget*					CentralWidget;

	QVBoxLayout*				MainLayout;
	QTreeWidget*				Connections;
	QInteractionWidget*			InteractionWidget;
	QRenderOutputWidget*		RenderOutputWidget;
	QTimer						Timer;
};