
#pragma once

#include <QMainWindow>
#include <QSettings>
#include <QVBoxLayout>
#include <QTimer>

#include "renderoutputwidget.h"

class QRenderer;

class QRendererWindow : public QMainWindow
{
    Q_OBJECT

public:
	QRendererWindow(QRenderer* Renderer, QWidget* Parent = 0, Qt::WindowFlags WindowFlags = 0);
	virtual ~QRendererWindow();

	void CreateStatusBar();

public slots:
	void OnTimer();

private:
	QSettings					Settings;
	QRenderer*					Renderer;
	QRenderOutputWidget*		RenderOutputWidget;
	QTimer						Timer;
};