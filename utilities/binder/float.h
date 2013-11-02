#ifndef QFloatBinder_H
#define QFloatBinder_H

#include "binder.h"

class EXPOSURE_RENDER_DLL QFloatBinder : public QBinder
{
    Q_OBJECT

public:
    QFloatBinder(QAttribute* Attribute, QWidget* Parent = 0);
    virtual ~QFloatBinder();

private:
	QPushButton*		ToMinimum;
	QPushButton*		Decrement;
	QSlider*			Slide;
	QDoubleSpinBox*		Edit;
	QPushButton*		Increment;
	QPushButton*		ToMaximum;
	QPushButton*		Reset;
};

#endif
