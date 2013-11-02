#ifndef QCameraWidget_H
#define QCameraWidget_H

#include "editwidget.h"
#include "api\camera.h"

class EXPOSURE_RENDER_DLL QCameraWidget : public QEditWidget
{
    Q_OBJECT

public:
    QCameraWidget(QWidget *parent = 0);
    virtual ~QCameraWidget();

public slots:
	void OnExport();
	void OnImport();

protected:
	QCamera			Camera;
	QPushButton*	Export;
	QPushButton*	Import;
private:
};

#endif
