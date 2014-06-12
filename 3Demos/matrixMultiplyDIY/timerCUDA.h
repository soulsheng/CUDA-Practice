
#include <cuda_runtime.h>
#include <cuda.h>

class timerCUDA
{
public:
	timerCUDA();
	~timerCUDA();
	void start();
	void stop();
	float getTime();

private:
	cudaEvent_t  tStart, tStop;
	float   differTime;
};