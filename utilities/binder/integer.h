#ifndef QIntegerBinder_H
#define QIntegerBinder_H

#include "binder.h"

class QIntegerAttribute;

class EXPOSURE_RENDER_DLL QIntegerBinder : public QBinder
{
    Q_OBJECT

public:
    QIntegerBinder(QAttribute* Attribute, const bool& Tracking = false, QWidget* Parent = 0);
    virtual ~QIntegerBinder();

	QIntegerAttribute* GetAttribute();

public slots:
	void OnValueChanged(int Value);
	void OnMinimumChanged(int Value);
	void OnMaximumChanged(int Value);
	void OnSliderValueChanged(int Value);
	void OnEditValueChanged(int Value);
	void OnResetValue();
	void OnToMinimum();
	void OnDecrement();
	void OnIncrement();
	void OnToMaximum();

private:
	QPushButton*		ToMinimum;
	QPushButton*		Decrement;
	QSlider*			Slide;
	QPushButton*		Increment;
	QPushButton*		ToMaximum;
	QSpinBox*			Edit;
	QPushButton*		Reset;
};

#endif
