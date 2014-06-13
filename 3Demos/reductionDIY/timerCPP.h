#include <Windows.h>

class timerC
{
public:
	timerC();
	void start();
	void stop();
	float getTime();

private:
	LARGE_INTEGER  freq;
	LARGE_INTEGER  tStart, tStop;
	float   differTime;
};