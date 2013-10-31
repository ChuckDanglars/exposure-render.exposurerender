
#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QtNetwork>

class QCompositorSocket;
class QInteractionWidget;
class QRenderOutputWidget;
class QPushButton;

class QGuiWindow : public QMainWindow
{
    Q_OBJECT

public:
	QGuiWindow(QCompositorSocket* CompositorSocket, QWidget* Parent = 0, Qt::WindowFlags WindowFlags = 0);
	virtual ~QGuiWindow();

	QRenderOutputWidget* GetRenderOutputWidget()
	{
		return this->RenderOutputWidget;
	}

public slots:
	void OnTimer();
	void OnUploadVolume();
	void OnUploadBitmap();
	void OnCameraUpdate(float* Position, float* FocalPoint, float* ViewUp);

private:
	QCompositorSocket*			CompositorSocket;
	QSettings					Settings;
	QWidget*					CentralWidget;
	QVBoxLayout*				MainLayout;
	QRenderOutputWidget*		RenderOutputWidget;
	QTimer						Timer;
	QPushButton*				UploadVolume;
	QPushButton*				UploadBitmap;
};