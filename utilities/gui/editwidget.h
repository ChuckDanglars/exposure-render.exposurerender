#ifndef QEditWidget_H
#define QEditWidget_H

#include "attribute\float.h"

class EXPOSURE_RENDER_DLL QEditWidget : public QWidget
{
    Q_OBJECT

public:
    QEditWidget(QWidget *parent = 0);
    virtual ~QEditWidget();

	void AddAttribute(QAttribute* Attribute);
	void Build();

private:
	QVBoxLayout*			Layout;
	QList<QAttribute*>		Attributes;
};

#endif
