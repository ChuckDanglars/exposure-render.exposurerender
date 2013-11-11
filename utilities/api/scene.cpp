
#include "scene.h"

QDataStream& operator << (QDataStream& Out, const QScene& Scene)
{
	Out << Scene.Camera;
	Out << Scene.Props;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QScene& Scene)
{
    In >> Scene.Camera;
	In >> Scene.Props;

    return In;
}

QScene::QScene(QObject* Parent /*= 0*/) :
	QObject(Parent)
{
}

QScene::~QScene()
{
}