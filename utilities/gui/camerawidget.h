#ifndef QCameraWidget_H
#define QCameraWidget_H

#include "editwidget.h"

class EXPOSURE_RENDER_DLL QCameraWidget : public QEditWidget
{
    Q_OBJECT

public:
    QCameraWidget(QWidget *parent = 0);
    virtual ~QCameraWidget();

private:
	QFloatAttribute			ApertureSize;
	QFloatAttribute			FieldOfView;
	QFloatAttribute			FocalDistance;
	QIntegerAttribute		FilmWidth;
	QIntegerAttribute		FilmHeight;
	QFloatAttribute			Exposure;
	QFloatAttribute			Gamma;
};

#endif
