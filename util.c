#include "util.h"
#include <time.h>
#include "math.h"

uint64_t getTimeMicroseconds64() {
  uint64_t nTime;
  struct timespec tSpec;

  clock_gettime(CLOCK_REALTIME, &tSpec);

  nTime = (uint64_t)tSpec.tv_sec * 1000000 + (uint64_t)tSpec.tv_nsec / 1000;
  return nTime;
}

float* transpose(float* weight, int h, int w) {
  float* new_weight = (float*)malloc(w * h * 4);
  int i, j;
  for (i = 0; i < w; ++i) {
    for (j = 0; j < h; ++j) {
      new_weight[j * w + i] = weight[i * h + j];
    }
  }

  free(weight);
  return new_weight;
}

float* get_parameter(const char* filename, int size) {
  float* parameter = (float*)malloc(size * 4);
  if (!parameter) {
    printf("Bad Malloc\n");
    exit(0);
  }
  FILE* ptr = fopen(filename, "rb");

  if (!ptr) {
    printf("Bad file path: %p, %s\n", ptr, strerror(errno));
    exit(0);
  }
  fread(parameter, size * 4, 1, ptr);

  fclose(ptr);
  return parameter;
}

float output_checker(float* A, float* B, int len, int channel, int shift) {
  int error_cnt = 0, i, j, k;
  float max_error = 0;
  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) {
      for (k = 0; k < channel; k++) {
        float diff = fabs(
            A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k] -
            B[(i * len + j) * channel + k]);
        if (diff > 1e-5)
          error_cnt++;
        if (diff > max_error)
          max_error = diff;
      }
    }
  }
  printf("[max_error: %f][error_cnt: %d]\n", max_error, error_cnt);
}
