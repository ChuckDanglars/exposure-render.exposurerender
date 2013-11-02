#ifndef QFloatBinder_H
#define QFloatBinder_H

#include "binder.h"

class QFloatAttribute;

class EXPOSURE_RENDER_DLL QFloatBinder : public QBinder
{
    Q_OBJECT

public:
    QFloatBinder(QAttribute* Attribute, const bool& Tracking = false, QWidget* Parent = 0);
    virtual ~QFloatBinder();

	QFloatAttribute* GetAttribute();

public slots:
	void OnValueChanged(float Value);
	void OnMinimumChanged(float Value);
	void OnMaximumChanged(float Value);
	void OnSliderValueChanged(int Value);
	void OnEditValueChanged(double Value);
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
	QDoubleSpinBox*		Edit;
	QPushButton*		Reset;
	int					FloatFactor;
};

#endif
