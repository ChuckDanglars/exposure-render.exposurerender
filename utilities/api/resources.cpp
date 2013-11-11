
#include "camera.h"

QDataStream& operator << (QDataStream& Out, const QCamera& Camera)
{
	Out << Camera.ApertureSize;
	Out << Camera.FieldOfView;
	Out << Camera.FocalDistance;
	Out << Camera.FilmWidth;
	Out << Camera.FilmHeight;
	Out << Camera.Exposure;
	Out << Camera.Gamma;

    return Out;
}

QDataStream& operator >> (QDataStream& In, QCamera& Camera)
{
	In >> Camera.ApertureSize;
	In >> Camera.FieldOfView;
	In >> Camera.FocalDistance;
	In >> Camera.FilmWidth;
	In >> Camera.FilmHeight;
	In >> Camera.Exposure;
	In >> Camera.Gamma;

    return In;
}

QCamera::QCamera(QObject* Parent /*= 0*/) :
	QObject(Parent),
	ApertureSize("Aperture size", "Lens opening size"),
	FieldOfView("FOV", "Field of view"),
	FocalDistance("Focal distance", "Focal distance"),
	FilmWidth("Width", "Film width", 1024, 1024, 0, 2048),
	FilmHeight("Height", "Film height", 768, 768, 0, 2048),
	Exposure("Exposure", "Film exposure"),
	Gamma("Gamma", "Film gamma")
{
}

QCamera::~QCamera()
{
}