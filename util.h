#ifndef __UTIL_H__
#define __UTIL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

float* get_parameter(const char* filename, int size);

float* transpose(float* weight, int h, int w);

uint64_t getTimeMicroseconds64();

float output_checker(float* A, float* B, int len, int channel, int shift);

#ifdef __cplusplus
}
#endif

#endif
