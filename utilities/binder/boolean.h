#ifndef QBooleanBinder_H
#define QBooleanBinder_H

#include "binder.h"

class QBooleanAttribute;

class EXPOSURE_RENDER_DLL QBooleanBinder : public QBinder
{
    Q_OBJECT

public:
    QBooleanBinder(QAttribute* Attribute, QWidget* Parent = 0);
    virtual ~QBooleanBinder();

	QBooleanAttribute* GetAttribute();

public slots:
	void OnValueChanged(bool Value);
	void OnCheckBoxStateChanged(int State);

private:
	QCheckBox*			CheckBox;
};

#endif
