#ifndef QOptionAttribute_H
#define QOptionAttribute_H

#include "attribute\integer.h"

class EXPOSURE_RENDER_DLL QOptionAttribute : public QIntegerAttribute
{
    Q_OBJECT

public:
    QOptionAttribute(const QString& Name, const QString& Description, const int& Value = 0, const int& DefaultValue = 0, const QStringList& Strings = QStringList(), QObject* Parent = 0);
    virtual ~QOptionAttribute();

	Q_PROPERTY(QStringList Strings READ GetStrings WRITE SetStrings NOTIFY StringsChanged)

	void SetStrings(const QStringList& Strings)				{ this->Strings = Strings; emit StringsChanged(this->Strings);									}
	QStringList GetStrings() const							{ return this->Strings;																			}
	
	void Initialize();

signals:
	void StringsChanged(QStringList);

protected:
	QStringList		Strings;

	friend QDataStream& operator << (QDataStream& Out, const QOptionAttribute& OptionAttribute);
	friend QDataStream& operator >> (QDataStream& In, QOptionAttribute& OptionAttribute);
};

QDataStream& operator << (QDataStream& Out, const QOptionAttribute& OptionAttribute);
QDataStream& operator >> (QDataStream& In, QOptionAttribute& OptionAttribute);

#endif
