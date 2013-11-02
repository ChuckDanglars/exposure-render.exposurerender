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

protected:
	QCamera		Camera;

private:
};

#endif
