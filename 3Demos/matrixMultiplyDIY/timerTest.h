#include <Windows.h>

class timerTest
{
public:
	timerTest();
	void start();
	virtual void stop();
	float getTime();

private:
	LARGE_INTEGER  freq;
	LARGE_INTEGER  tStart, tStop;
	float   differTime;
};

class timerTestCU : public timerTest
{
public:
	void stop();
};