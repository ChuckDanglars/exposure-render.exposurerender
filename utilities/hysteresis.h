#pragma once

#include <QObject>
#include <QList>

#define MAX_TIMINGS 30

class QHysteresis : public QObject
{
    Q_OBJECT

public:
	QHysteresis(QObject* Parent = 0);
	virtual ~QHysteresis() {};
	
	void PushValue(const float& Value);
	float GetAverageValue();

protected:
	void ComputeAverage();

private:
	QList<float>	Values;
	float			AverageValue;
};