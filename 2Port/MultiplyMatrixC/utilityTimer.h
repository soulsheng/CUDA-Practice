
#include <windows.h>

class TimerCPU
{
public:
	TimerCPU();
	void start();
	void stop();
	float getTime();
protected:
private:
	double  freq;
	//! Start of measurement
	LARGE_INTEGER  start_time;
	//! End of measurement
	LARGE_INTEGER  end_time;

	//! Time difference between the last start and stop
	float  diff_time;
};

class TimerGPU
{
public:
	void start();
	void stop();
	float getTime();
protected:
private:
	float  diff_time;

};

class TimerXPU
{
public:
	void start();
	void stop();
	float getTime();
protected:
private:
	float  diff_time;

};
