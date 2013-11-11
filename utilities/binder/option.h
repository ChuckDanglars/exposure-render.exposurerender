#ifndef QOptionBinder_H
#define QOptionBinder_H

#include "binder.h"

class QOptionAttribute;

class EXPOSURE_RENDER_DLL QOptionBinder : public QBinder
{
    Q_OBJECT

public:
    QOptionBinder(QAttribute* Attribute, QWidget* Parent = 0);
    virtual ~QOptionBinder();

	QOptionAttribute* GetAttribute();

public slots:
	void OnValueChanged(int Value);
	void OnComboBoxIndexValueChanged(int Value);
	void OnFirst();
	void OnPrevious();
	void OnNext();
	void OnLast();
	void OnResetValue();

private:
	QComboBox*			ComboBox;
	QPushButton*		First;
	QPushButton*		Previous;
	QPushButton*		Next;
	QPushButton*		Last;
	QPushButton*		Reset;
};

#endif
