#ifndef __UTIL_H__
#define __UTIL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <errno.h>


float *get_parameter(const char *filename, int size);
uint64_t getTimeMicroseconds64();
float output_checker(float *A, float *B, int len, int channel, int shift);

#ifdef __cplusplus
}
#endif

#endif
