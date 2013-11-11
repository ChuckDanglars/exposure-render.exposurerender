#ifndef QScene_H
#define QScene_H

#include "camera.h"

class EXPOSURE_RENDER_DLL QScene : public QObject
{
    Q_OBJECT

public:
    QScene(QObject* Parent = 0);
    virtual ~QScene();

	QCamera				Camera;
	QList<QString>		Props;
};

QDataStream &operator << (QDataStream& Out, const QScene& Scene);
QDataStream &operator >> (QDataStream& In, QScene& Scene);

#endif
