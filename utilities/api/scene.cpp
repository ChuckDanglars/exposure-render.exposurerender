
#include "scene.h"

QDataStream& operator << (QDataStream& Out, const QScene& Scene)
{
	Out << Scene.Camera;
	Out << Scene.Volumes;
	Out << Scene.Lights;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QScene& Scene)
{
    In >> Scene.Camera;
	In >> Scene.Volumes;
	In >> Scene.Lights;

    return In;
}

QScene::QScene(QObject* Parent /*= 0*/) :
	QObject(Parent)
{
}

QScene::~QScene()
{
}