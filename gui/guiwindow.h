
#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QtNetwork>

class QClientSocket;
class QInteractionWidget;
class QRenderOutputWidget;

class QGuiWindow : public QMainWindow
{
    Q_OBJECT

public:
	QGuiWindow(QClientSocket* ClientSocket, QWidget* Parent = 0, Qt::WindowFlags WindowFlags = 0);
	virtual ~QGuiWindow();

	void CreateStatusBar();
	
	QRenderOutputWidget* GetRenderOutputWidget()
	{
		return this->RenderOutputWidget;
	}

public slots:
	void OnTimer();

private:
	QClientSocket*				ClientSocket;
	QSettings					Settings;
	QWidget*					CentralWidget;
	QVBoxLayout*				MainLayout;
	QRenderOutputWidget*		RenderOutputWidget;
	QTimer						Timer;
};