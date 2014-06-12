
#include "timerCPP.h"
#include <cuda_runtime.h>

	timerTest::timerTest()
	{
		QueryPerformanceFrequency( &freq ) ;
	}

	void timerTest::start()
	{
		QueryPerformanceCounter( &tStart ) ;
	}

	void timerTest::stop()
	{
		QueryPerformanceCounter( &tStop ) ;	
	}

	float timerTest::getTime()
	{
		differTime = (tStop.QuadPart - tStart.QuadPart)/(float)freq.QuadPart ;
		return differTime*1000.0f ;
	}

	void timerTestCU::stop()
	{
		cudaDeviceSynchronize();
		timerTest::stop();
	}
