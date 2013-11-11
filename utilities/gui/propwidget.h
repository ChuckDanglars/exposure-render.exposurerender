#ifndef QCameraWidget_H
#define QCameraWidget_H

#include "editwidget.h"
#include "api\prop.h"

class EXPOSURE_RENDER_DLL QPropWidget : public QEditWidget
{
    Q_OBJECT

public:
    QPropWidget(QProp* Prop, QWidget *parent = 0);
    virtual ~QPropWidget();

public slots:
	void OnExport();
	void OnImport();

protected:
	QProp*			Prop;
	QPushButton*	Export;
	QPushButton*	Import;
private:
};

#endif
