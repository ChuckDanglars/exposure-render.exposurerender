#ifndef QBind_H
#define QBind_H

#include "attribute\attribute.h"

class QAttribute;

class EXPOSURE_RENDER_DLL QBinder : public QObject
{
    Q_OBJECT

public:
    QBinder(QAttribute* Attribute, QObject* Parent = 0);
    virtual ~QBinder();

	QWidget* GetWidget() { return this->Widget; };

protected:
	QHBoxLayout*	Layout;

private:
	QAttribute*		Attribute;
	QWidget*		Widget;
	QLabel*			Label;
};

#endif
