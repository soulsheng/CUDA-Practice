#ifndef _TRANSPOSEHEADER_H_
#define _TRANSPOSEHEADER_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include<Windows.h>
#pragma comment(lib,"winmm.lib")

// Thread block size
#define BLOCK_SIZE 16

// matrix size
#define width 1024
#define height 1024

#define NUM_REPS  100

#endif // _TRANSPOSEHEADER_H_