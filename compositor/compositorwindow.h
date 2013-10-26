
#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QtNetwork>

#include "clientsocket.h"

class QServer;
class QInteractionWidget;
class QLabel;
class QRenderOutputWidget;

class QConnectionItem : public QObject
{
	Q_OBJECT

public:
	QConnectionItem(QClientSocket* ClientSocket, QTreeWidget* TreeWidget, QObject* Parent = 0) :
		QObject(Parent),
		ClientSocket(0),
		TreeWidget(0)
	{
		this->ClientSocket		= ClientSocket;
		this->TreeWidget		= TreeWidget;
		this->TreeWidgetItem	= new QTreeWidgetItem();

		connect(this->ClientSocket, SIGNAL(UpdateFps(const float&)), this, SLOT(OnUpdateFps(const float&)));
		connect(this->ClientSocket, SIGNAL(UpdateJpgNoBytes(const int&)), this, SLOT(OnUpdateJpgNoBytes(const int&)));
		connect(this->ClientSocket, SIGNAL(UpdateJpgEncodeTime(const float&)), this, SLOT(OnUpdateJpgEncodeTime(const float&)));
		connect(this->ClientSocket, SIGNAL(UpdateJpgDecodeTime(const float&)), this, SLOT(OnUpdateJpgDecodeTime(const float&)));
		
		this->TreeWidget->invisibleRootItem()->addChild(this->TreeWidgetItem);

		this->TreeWidgetItem->setTextAlignment(2, Qt::AlignRight);
		this->TreeWidgetItem->setTextAlignment(3, Qt::AlignRight);
		this->TreeWidgetItem->setTextAlignment(4, Qt::AlignRight);
		this->TreeWidgetItem->setTextAlignment(5, Qt::AlignRight);
	}

signals:
	void remove(QConnectionItem*);

public slots:
	void OnSocketConnected()
	{
		this->Host = this->ClientSocket->localAddress().toString();
		this->TreeWidgetItem->setText(0, this->Host);
		this->TreeWidgetItem->setText(1, "connected");
		this->TreeWidgetItem->setText(6, "<undefined>");
	}

	void OnSocketDisconnected()
	{
		this->TreeWidgetItem->setText(1, "disconnected");
		this->TreeWidgetItem->setText(2, "<undefined>");
		this->TreeWidgetItem->setText(3, "<undefined>");
		this->TreeWidgetItem->setText(4, "<undefined>");
		this->TreeWidgetItem->setText(5, "<undefined>");
	}

	void OnUpdateJpgNoBytes(const int& JpgNoBytes)
	{
		QString Text;
		Text.sprintf("%0.3f KB", (float)JpgNoBytes / 1024.0f);
		this->TreeWidgetItem->setText(2, Text);
	}

	void OnUpdateFps(const float& AverageFps)
	{
		QString Text;
		Text.sprintf("%0.2f fps", AverageFps);
		this->TreeWidgetItem->setText(3, Text);
	}

	void OnUpdateJpgEncodeTime(const float& JpgEncodeTime)
	{
		QString Text;
		Text.sprintf("%0.2f ms", JpgEncodeTime);
		this->TreeWidgetItem->setText(4, Text);
	}

	void OnUpdateJpgDecodeTime(const float& JpgDecodeTime)
	{
		QString Text;
		Text.sprintf("%0.2f ms", JpgDecodeTime);
		this->TreeWidgetItem->setText(5, Text);
	}

private:
	QClientSocket*		ClientSocket;
	QTreeWidget*		TreeWidget;
	QTreeWidgetItem*	TreeWidgetItem;
	QString				Host;
};

class QCompositorWindow : public QMainWindow
{
    Q_OBJECT

public:
	QCompositorWindow(QServer* Server, QWidget* Parent = 0, Qt::WindowFlags WindowFlags = 0);
	virtual ~QCompositorWindow();

	void CreateStatusBar();
	QRenderOutputWidget* GetRenderOutputWidget() { return this->RenderOutputWidget; }

public slots:
	void removeConnectionItem(QConnectionItem* ConnectionItem);
	void newThreadedSocket(QClientSocket*);
	void OnUpdateEstimateIntegration(const float& EstimateIntegration);
	void OnTimer();

private:
	QSettings					Settings;
	QWidget*					CentralWidget;
	QServer*					Server;
	QVBoxLayout*				MainLayout;
	QTreeWidget*				Connections;
	QInteractionWidget*			InteractionWidget;
	QList<QConnectionItem*>		ConnectionItems;
	QLabel*						AvgCombine;
	QRenderOutputWidget*		RenderOutputWidget;
	QTimer						Timer;
};