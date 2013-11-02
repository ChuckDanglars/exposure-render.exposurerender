#ifndef QEditWidget_H
#define QEditWidget_H

#include "attribute\attributes.h"
#include "binder\binders.h"

class EXPOSURE_RENDER_DLL QEditWidget : public QWidget
{
    Q_OBJECT

public:
    QEditWidget(QWidget *parent = 0);
    virtual ~QEditWidget();

private:
	QVBoxLayout*			Layout;
};

#endif
