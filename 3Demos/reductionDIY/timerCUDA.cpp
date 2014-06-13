
#include "timerCUDA.h"

	timerCUDA::timerCUDA()
	{
		cudaEventCreate( &tStart );
		cudaEventCreate( &tStop );
		differTime = 1.0f;
	}
	timerCUDA::~timerCUDA()
	{
		cudaEventDestroy( tStart );
		cudaEventDestroy( tStop );
	}
	void timerCUDA::start()
	{
		cudaEventRecord( tStart,0 ) ;
	}

	void timerCUDA::stop()
	{
		cudaEventRecord( tStop,0 ) ;	
	}

	float timerCUDA::getTime()
	{
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&differTime, tStart, tStop);
		
		return differTime ;
	}
