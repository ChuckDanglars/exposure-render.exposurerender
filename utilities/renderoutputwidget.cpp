
#include <QtGui>
#include <QtOpenGL>

#include <math.h>

#include "renderoutputwidget.h"

using namespace ExposureRender;

QRenderOutputWidget::QRenderOutputWidget(QWidget* Parent) :
	QGLWidget(Parent),
	Image(),
	AspectRatio(1.0f),
	LastPos(),
	TextureID(0),
	Position(1.0f),
	FocalPoint(0.0f),
	ViewUp(0.0f, 1.0f, 0.0f),
	Margin(10.0f)
{
}

QRenderOutputWidget::~QRenderOutputWidget()
{
}

void QRenderOutputWidget::initializeGL()
{
	glClearColor(0.2f, 0.2f, 0.2f, 0.0f);

	glGenTextures(1, &this->TextureID);
	glBindTexture(GL_TEXTURE_2D, this->TextureID);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void QRenderOutputWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, this->TextureID);
	
	glTexImage2D(GL_TEXTURE_2D, 0, 3, this->Image.Width(), this->Image.Height(), 0, GL_RGB, GL_UNSIGNED_BYTE, (unsigned char*)this->Image.GetData());

	this->ComputeQuad();

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(this->Quad[0][0], this->Quad[1][0], 0.0);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(this->Quad[0][1], this->Quad[1][0], 0.0);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(this->Quad[0][1], this->Quad[1][1], 0.0);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(this->Quad[0][0], this->Quad[1][1], 0.0);
	glEnd();
}

void QRenderOutputWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1.0l, 1.0l);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void QRenderOutputWidget::mousePressEvent(QMouseEvent *event)
{
    this->LastPos = event->pos();
}

void QRenderOutputWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - this->LastPos.x();
    int dy = event->y() - this->LastPos.y();

    if (event->buttons() & Qt::LeftButton)
	{
		this->Orbit(-dy, -dx);
    }

	if (event->buttons() & Qt::MiddleButton)
	{
		this->Pan(dy, -dx);
	}

	if (event->buttons() & Qt::RightButton)
	{
		this->Zoom(dy);
	}

    this->LastPos = event->pos();

	emit CameraUpdate(this->Position.D, this->FocalPoint.D, this->ViewUp.D);
}

void QRenderOutputWidget::SetImage(HostBuffer2D<ColorRGBuc>& Image)
{
	this->Image = Image;
	
	this->updateGL();
}

void QRenderOutputWidget::ComputeQuad()
{
	float AspectRatioWindow = (float)this->height() / (float)this->width();

	float QuadSize[2] = { 0.0f };

	if (this->Image.AspectRatio() > AspectRatioWindow)
	{
		QuadSize[1] = this->height();
		QuadSize[0] = QuadSize[1] / this->Image.AspectRatio();
	}
	else
	{
		QuadSize[0] = this->width();
		QuadSize[1] = QuadSize[0] * this->Image.AspectRatio();
	}

	QuadSize[0] -= 2.0f * this->Margin;
	QuadSize[1] -= 2.0f * this->Margin;

	this->Quad[0][0] = 0.5f * (this->width() - QuadSize[0]);
	this->Quad[0][1] = this->Quad[0][0] + QuadSize[0];
	this->Quad[1][0] = 0.5f * (this->height() - QuadSize[1]);
	this->Quad[1][1] = this->Quad[1][0] + QuadSize[1];
}

void QRenderOutputWidget::Zoom(float amount)
{
	Vec3f reverseLoS = Position - FocalPoint;

	if (amount > 0)
	{	
		reverseLoS = reverseLoS * 1.1f;
	}
	else if (amount < 0)
	{	
		if (reverseLoS.Length() > 0.0005f)
		{ 
			reverseLoS = reverseLoS * 0.9f;
		}
	}

	Position = reverseLoS + FocalPoint;
}

void QRenderOutputWidget::Pan(float DownDegrees, float RightDegrees)
{
	Vec3f LoS = FocalPoint - Position;

	Vec3f right		= LoS.Cross(ViewUp);
	Vec3f orthogUp	= LoS.Cross(right);

	right.Normalize();
	orthogUp.Normalize();

	const float Length = (FocalPoint - Position).Length();

	const unsigned int WindowWidth	= this->width();

	const float U = Length * (RightDegrees / WindowWidth);
	const float V = Length * (DownDegrees / WindowWidth);

	Position	= Position + right * U - ViewUp * V;
	FocalPoint	= FocalPoint + right * U - ViewUp * V;
}

void QRenderOutputWidget::Orbit(float DownDegrees, float RightDegrees)
{
	Vec3f ReverseLoS = Position - FocalPoint;

	Vec3f right		= ViewUp.Cross(ReverseLoS);
	Vec3f orthogUp	= ReverseLoS.Cross(right);
	Vec3f Up = Vec3f(0.0f, 1.0f, 0.0f);
		
	ReverseLoS.RotateAxis(right, DownDegrees);
	ReverseLoS.RotateAxis(Up, RightDegrees);
	ViewUp.RotateAxis(right, DownDegrees);
	ViewUp.RotateAxis(Up, RightDegrees);

	Position = ReverseLoS + FocalPoint;
}