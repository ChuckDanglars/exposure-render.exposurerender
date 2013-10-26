#pragma once

#include <QTcpSocket>
#include <QDebug>

class QBaseSocket : public QTcpSocket
{
    Q_OBJECT

public:
    QBaseSocket(QObject* Parent = 0);
	virtual ~QBaseSocket();

public:
	virtual void OnData(const QString& Action, QDataStream& DataStream);

public slots:
	void OnReadyRead();

private:
	quint32		BlockSize;
};