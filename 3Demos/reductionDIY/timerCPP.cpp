
#include "timerCPP.h"

	timerC::timerC()
	{
		QueryPerformanceFrequency( &freq ) ;
	}

	void timerC::start()
	{
		QueryPerformanceCounter( &tStart ) ;
	}

	void timerC::stop()
	{
		QueryPerformanceCounter( &tStop ) ;	
	}

	float timerC::getTime()
	{
		differTime = (tStop.QuadPart - tStart.QuadPart)/(float)freq.QuadPart ;
		return differTime*1000.0f ;
	}
