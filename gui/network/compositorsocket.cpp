
#include "compositorsocket.h"

QCompositorSocket::QCompositorSocket(QObject* Parent /*= 0*/) :
	QBaseSocket(Parent),
	Settings("gui.ini", QSettings::IniFormat)
{
}

QCompositorSocket::~QCompositorSocket()
{
}