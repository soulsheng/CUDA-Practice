
#include "utilityTimer.h"

void TimerCPU::start()
{
	QueryPerformanceCounter(&start_time);

}

void TimerCPU::stop()
{
	QueryPerformanceCounter(&end_time);
}

float TimerCPU::getTime()
{
	diff_time = (float)
		(((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);
	return diff_time;
}

TimerCPU::TimerCPU()
{
	LARGE_INTEGER temp;

	// get the tick frequency from the OS
	QueryPerformanceFrequency(&temp);

	// convert to type in which it is needed
	freq = ((double) temp.QuadPart) / 1000.0;
}

void TimerGPU::start()
{

}

void TimerGPU::stop()
{
	
}

float TimerGPU::getTime()
{

	return diff_time;
}

void TimerXPU::start()
{

}

void TimerXPU::stop()
{

}

float TimerXPU::getTime()
{

	return diff_time;
}
