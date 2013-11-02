#ifndef QCamera_H
#define QCamera_H

#include "attribute\attributes.h"

class EXPOSURE_RENDER_DLL QCamera : public QObject
{
    Q_OBJECT

public:
    QCamera(QObject* Parent = 0);
    virtual ~QCamera();

	QFloatAttribute			ApertureSize;
	QFloatAttribute			FieldOfView;
	QFloatAttribute			FocalDistance;
	QIntegerAttribute		FilmWidth;
	QIntegerAttribute		FilmHeight;
	QFloatAttribute			Exposure;
	QFloatAttribute			Gamma;
};

QDataStream& operator << (QDataStream& Out, const QCamera& Camera);
QDataStream& operator >> (QDataStream& In, QCamera& Camera);

#endif
