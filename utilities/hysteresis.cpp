
#include "hysteresis.h"

QHysteresis::QHysteresis(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Values(),
	AverageValue(0.0f)
{
}

void QHysteresis::PushValue(const float& Timing)
{
	this->Values.append(Timing);
	
	if (this->Values.size() >= MAX_TIMINGS)
	{
		this->Values.takeFirst();
	}

	this->ComputeAverage();
}

void QHysteresis::ComputeAverage()
{
	this->AverageValue = 0.0f;

	foreach (float Value, this->Values)
	{
		this->AverageValue += Value;
	}

	this->AverageValue /= (float)this->Values.size();
}

float QHysteresis::GetAverageValue()
{
	return this->AverageValue;
}